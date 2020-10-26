"""
https://docs.scrapy.org/en/latest/intro/tutorial.html
"""

import scrapy

class LoveSenseSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://matchlessdaily.com/love-songs-lyrics-for-your-boyfriend/',
            'https://www.yourtango.com/2018318763/best-love-quotes-song-lyrics-have-romantic-meanings',
            'https://www.scriptslug.com/script/working-girl-1988',
            'https://www.washingtonpost.com/',
            'https://academic.oup.com',
            'https://www.fbi.gov/video-repository/fbi-statement-on-the-arrest-of-former-uber-cso-for-covering-up-2016-hack/view',
            'https://www.scamwatch.gov.au/types-of-scams/dating-romance',
            'https://www.ic3.gov/media/2011/110429.aspx',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'lovesens-datalake-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)