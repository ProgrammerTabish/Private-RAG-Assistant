import json,io,re
from pathlib import Path
from urllib.parse import unquote
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader,PdfWriter
P=Path('documents/spg_compliance')
ms=json.loads((P/'manifest.json').read_text('utf-8'))
s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0'
def fetch(u):
    r=s.get(u,timeout=60); r.raise_for_status(); return r
def save(n,u):
    m=next(x for x in ms if x['number']==n)
    r=fetch(u)
    assert r.content.startswith(b'%PDF'), 'Response is not PDF'
    reader=PdfReader(io.BytesIO(r.content))
    (P/m['filename']).write_bytes(r.content)
    m.update(status='Downloaded',download_url=r.url,pages=len(reader.pages)); m.pop('error',None)
    print(n,'Downloaded',len(reader.pages),flush=True)
    (P/'manifest.json').write_text(json.dumps(ms,ensure_ascii=False,indent=2),encoding='utf-8')
for n in [3,4,5,23]:
    try:
        if n==3: u='https://bafin.de/SharedDocs/Downloads/DE/Auslegungsentscheidung/dl_ae_aua_bt_ki_gw.pdf?__blob=publicationFile&v=6'
        elif n==4:
            u='https://www.eba.europa.eu/sites/default/files/document_library/Publications/Guidelines/2023/EBA-GL-2023-03/1061654/Guidelines%20ML%20TF%20Risk%20Factors_conslidated.pdf.pdf'
        else:
            links=json.loads((P/f'{n:02d}_links.json').read_text('utf-8'))
            u=next(a['url'] for a in links if re.search(r'_DE(?:[_. %])',unquote(a['url'])))
        save(n,u)
        if n==4:
            amendment=next(a['url'] for a in json.loads((P/'04_links.json').read_text('utf-8')) if '_DE_COR.pdf' in a['url'])
            r=fetch(amendment)
            w=PdfWriter(); m=next(x for x in ms if x['number']==4)
            w.append(P/m['filename']); w.append(PdfReader(io.BytesIO(r.content)))
            w.write(P/m['filename']); m['pages']=len(w.pages)
            m['version_note']='2023 consolidated English guidelines plus the final 2024 German amendment; no single 2024 consolidated PDF was found on the supplied page.'
            m['additional_download_url']=r.url
    except Exception as e: print(n,str(e),flush=True)
for m in ms:
    if 'eur-lex.europa.eu' in m['source_url']:
        m['error']='EUR-Lex browser access was denied; automated requests returned a JavaScript verification page.'
(P/'manifest.json').write_text(json.dumps(ms,ensure_ascii=False,indent=2),encoding='utf-8')
r=fetch(next(x['source_url'] for x in ms if x['number']==11))
r.encoding='utf-8'
(P/'11_source.html').write_text(r.text,encoding='utf-8')
soup=BeautifulSoup(r.text,'html.parser')
main=soup.find('main') or soup.find(id='content')
(P/'11_article.txt').write_text(main.get_text('\n',strip=True),encoding='utf-8')
print('Article saved',flush=True)
