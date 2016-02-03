from urllib3 import Request, urlopen
from urllib import urlencode, quote_plus

url = 'http://api.seibro.or.kr/openapi/service/StockSvc/getNewDepoSecnList'
queryParams = '?' + urlencode({ quote_plus('ServiceKey') : '서비스키', quote_plus('yyyymm') : '201301', quote_plus('searchType') : '2', quote_plus('issucoCustno') : '8424', quote_plus('numOfRows') : '999', quote_plus('pageNo') : '1' })

request = Request(url + queryParams)
request.get_method = lambda: 'GET'
response_body = urlopen(request).read()
print (response_body)