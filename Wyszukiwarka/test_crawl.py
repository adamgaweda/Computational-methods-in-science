import requests
from bs4 import BeautifulSoup

s = requests.Session()
s.headers['User-Agent'] = 'MOwNiT-SearchEngine/1.0 (university project)'
r = s.get('https://simple.wikipedia.org/wiki/Special:AllPages')
print(f"Status: {r.status_code}")
soup = BeautifulSoup(r.text, 'html.parser')
items = soup.select('ul.mw-allpages-chunk li')
print(f"Items on page: {len(items)}")
for li in items[:5]:
    a = li.find('a')
    if a:
        print(f"  {a.text}")
