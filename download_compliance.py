import json, re, io, time, sys
from pathlib import Path
from urllib.parse import urljoin, unquote
import requests, openpyxl
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT=Path(__file__).parent
OUT=ROOT/'documents'/'spg_compliance'
OUT.mkdir(exist_ok=True)
session=requests.Session()
session.headers['User-Agent']='Mozilla/5.0'
rows=[r for r in openpyxl.load_workbook(ROOT/'documents/EN_SPG_Banking_Compliance_Download_List_50.xlsx').active.values if isinstance(r[0],int)]
manifest=json.loads((OUT/'manifest.json').read_text('utf-8')) if (OUT/'manifest.json').exists() else []
def persist():
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
def get(url):
    r=session.get(url,timeout=55); r.raise_for_status(); return r
def save_pdf(r,entry):
    if not r.content.lstrip().startswith(b'%PDF'): return False
    reader=PdfReader(io.BytesIO(r.content))
    if not len(reader.pages): return False
    (OUT/entry['filename']).write_bytes(r.content)
    entry.update(status='Downloaded',download_url=r.url,pages=len(reader.pages))
    return True
for row in rows:
    n,_,priority,topic,title,_,kind,lang,note,filename,publisher,reference,url,fallback,*_=row
    previous=next((m for m in manifest if m['number']==n),None)
    if previous and previous['status']=='Downloaded': continue
    entry=dict(number=n,title=title,filename=filename,source_url=url,language=lang,note=note,reference=reference,status='Pending',attempts=[],candidates=[])
    if previous: manifest.remove(previous)
    manifest.append(entry); manifest.sort(key=lambda m:m['number'])
    print(f'{n:02d}/50 {title}',flush=True)
    try:
        r=get(url)
        if save_pdf(r,entry): persist(); print('  Downloaded',entry['pages'],'pages',flush=True); continue
        soup=BeautifulSoup(r.text,'html.parser')
        links=[dict(url=urljoin(r.url,a['href']),text=a.get_text(' ',strip=True)) for a in soup.select('a[href]')]
        candidates=[]
        if 'eur-lex.europa.eu' in url:
            cons=[]
            for a in links:
                u=unquote(a['url'])
                match=re.search(r'(0\d{4}[RL]\d{4}-\d{8})',u)
                if match and match[1][-8:]<='20261008': cons.append(match[1])
            if cons:
                celex=max(cons,key=lambda x:x[-8:]); entry['selected_version']=celex
                candidates.append('https://eur-lex.europa.eu/legal-content/DE/TXT/PDF/?uri=CELEX:'+celex)
            else:
                for a in links:
                    if '/PDF/' in a['url'] and ('/DE/' in a['url'] or '/de/' in a['url']): candidates.append(a['url'])
                uri=re.search(r'uri=([^&]+)',url)
                if uri: candidates.append('https://eur-lex.europa.eu/legal-content/DE/TXT/PDF/?uri='+uri[1])
                else: candidates.append(url.split('?')[0].replace('/oj/deu','/oj').rstrip('/')+'/deu/pdf')
        else:
            pdf_links=[a for a in links if '.pdf' in a['url'].lower() or '__blob=publicationFile' in a['url']]
            def score(a):
                t=unquote(a['url']+' '+a['text']).lower()
                value=0
                if 'gesetze-im-internet' in url: return 100 if '.pdf' in t else 0
                for term in ['guideline','leitlinie','final','2018','german','deutsch','_de.',' de ','aua','auslegung','consolidated']:
                    if term in t: value+=3
                for term in ['consultation','compliance table','compliance_table','assessment','feedback','factsheet']: 
                    if term in t: value-=20
                return value
            candidates.extend(a['url'] for a in sorted(pdf_links,key=score,reverse=True))
        entry['candidates']=list(dict.fromkeys(candidates))
        (OUT/f'{n:02d}_links.json').write_text(json.dumps(links,ensure_ascii=False,indent=2),encoding='utf-8')
        for target in entry['candidates'][:8]:
            try:
                pr=get(target)
                if save_pdf(pr,entry): break
                entry['attempts'].append({'url':target,'error':'Not a PDF'})
            except Exception as e: entry['attempts'].append({'url':target,'error':str(e)})
        if entry['status']!='Downloaded': entry['status']='Needs follow-up'; entry['error']='No verified PDF downloaded from source page'
    except Exception as e: entry.update(status='Needs follow-up',error=str(e))
    persist(); print(' ',entry['status'],entry.get('pages',entry.get('error','')),flush=True)
persist()
print('Downloaded',sum(m['status']=='Downloaded' for m in manifest),'of 50',flush=True)
