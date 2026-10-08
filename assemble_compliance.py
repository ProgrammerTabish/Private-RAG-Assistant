import json, io, html
from pathlib import Path
from bs4 import BeautifulSoup
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from pypdf import PdfReader,PdfWriter
import fitz
P=Path('documents/spg_compliance'); O=Path('output/pdf'); O.mkdir(parents=True,exist_ok=True)
ms=json.loads((P/'manifest.json').read_text('utf-8'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Entry',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallSource',fontSize=8,leading=11,textColor=colors.HexColor('#475569'),wordWrap='CJK',spaceAfter=8))
def para(t,style='Entry'): return Paragraph(html.escape(t),styles[style])
def footer(c,d):
    c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#64748b')); c.drawString(42,25,'SPG banking compliance | 8 October 2026'); c.drawRightString(553,25,str(d.page))
# Convert the complete official circular body, omitting site navigation and feedback controls.
m=next(x for x in ms if x['number']==11)
soup=BeautifulSoup((P/'11_source.html').read_text('utf-8'),'html.parser')
main=soup.find('main')
story=[para('Rundschreiben 3/2017 (GW) - Videoidentifizierungsverfahren','Title'),para('Official BaFin page converted to PDF. Publication date: 10 April 2017.','SmallSource'),para(m['source_url'],'SmallSource')]
started=False
for el in main.find_all(['h1','h2','h3','h4','p','ul','ol','table']):
    if el.find_parent(['p','ul','ol','table']): continue
    t=el.get_text(' ',strip=True)
    if t.startswith('Das Rundschreiben richtet sich'): started=True
    if started:
        if t in ['Weitere Informationen','Mehr zum Thema'] or t.startswith('Möchten Sie uns Feedback'): break
        if t: story.append(para(t,'Heading2' if el.name.startswith('h') else 'Entry'))
assert started
SimpleDocTemplate(str(P/m['filename']),rightMargin=42,leftMargin=42,topMargin=44,bottomMargin=44).build(story,onFirstPage=footer,onLaterPages=footer)
m.update(status='Downloaded',download_url=m['source_url'],pages=len(PdfReader(P/m['filename']).pages),version_note='Official HTML circular converted to PDF; published 10 April 2017.')
included=[x for x in ms if x['status']=='Downloaded']; missing=[x for x in ms if x['status']!='Downloaded']
front=[para('SPG banking compliance','Title'),para('Combined document collection','Heading1'),para('Retrieved 8 October 2026'),para(f'{len(included)} of 50 entries included. {len(missing)} entries could not be retrieved.'),para('Documents follow the original workbook order. PDF bookmarks open each included document. Source layouts and page numbering are preserved.'),para('Missing documents','Heading2'),para('EUR-Lex access was denied after its websites returned JavaScript verification pages. The BaFin Special Part for credit institutions returned HTTP 404. These documents are omitted; this collection is incomplete.'),para('Version note','Heading2'),para('Entry 4 combines the 2023 consolidated English risk-factor guidelines and the final 2024 German amendment. A single consolidated 2024 file was not available on the supplied page.'),PageBreak(),para('Download status and sources','Title')]
for x in ms:
    status='Included' if x['status']=='Downloaded' else 'MISSING'
    block=[para(f"{x['number']:02d}. {x['title']} - {status}"),para(x.get('version_note') or str(x.get('reference') or ''),'SmallSource'),para(x.get('download_url') or x['source_url'],'SmallSource')]
    front.append(KeepTogether(block)); front.append(Spacer(1,4))
buffer=io.BytesIO()
SimpleDocTemplate(buffer,rightMargin=42,leftMargin=42,topMargin=44,bottomMargin=44).build(front,onFirstPage=footer,onLaterPages=footer)
w=PdfWriter(); w.append(PdfReader(buffer)); w.add_outline_item('Download status and sources',1)
starts=[]
for x in included:
    start=len(w.pages); reader=PdfReader(P/x['filename'])
    w.append(reader,import_outline=False); w.add_outline_item(f"{x['number']:02d}. {x['title']}",start)
    x['combined_pdf_start_page']=start+1; starts.append(start)
w.add_metadata({'/Title':'SPG Banking Compliance - Combined Collection','/Subject':f'{len(included)} of 50 entries; incomplete collection','/Author':'SPG document collection'})
target=O/'SPG_Banking_Compliance_Combined.pdf'; w.write(target)
(P/'manifest.json').write_text(json.dumps(ms,ensure_ascii=False,indent=2),encoding='utf-8')
doc=fitz.open(target)
assert len(doc)==sum(x['pages'] for x in included)+len(PdfReader(buffer).pages)
qa=Path('tmp/pdfs'); qa.mkdir(parents=True,exist_ok=True)
for n in [0,1,starts[0],starts[3],starts[-1]]:
    doc[n].get_pixmap(matrix=fitz.Matrix(1,1)).save(qa/f'combined_page_{n+1}.png')
print(json.dumps({'included':len(included),'missing':[x['number'] for x in missing],'pages':len(doc),'bytes':target.stat().st_size,'file':str(target.resolve())}))
