from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import time
import os


class spider():
    def __init__(self):
        # 这个是一个用来控制chrome以无界面模式打开的浏览器
        # 创建一个参数对象，用来控制chrome以无界面的方式打开
        chrome_options = Options()  # 后面的两个是固定写法 必须这么写
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        
        """
        chromedriver download from : https://chromedriver.chromium.org/downloads
        """

        chrome_driver = 'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'
        if not os.path.exists('C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'):
            assert "please download chromedriver first"
        self.driver = webdriver.Chrome(executable_path= chrome_driver,options=chrome_options)

        # 想观看浏览器运行情况就打开这里
        # self.driver=webdriver.Chrome()

        self.wait = WebDriverWait(self.driver, 5)
        # 打开想要跳转的界面，此步不可缺少，不然会报错
        self.driver.get("http://star.iecity.com/China/0/zy2zp0")

    def download(self, url, name):
        response = requests.get(url=url)
        try:
            with open(f'images/{name}.jpg', 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"error:{e}")

    def parse(self):
        ul = self.driver.find_element_by_xpath('//*[@id="Main"]/div[4]/ul')
        lis = ul.find_elements_by_xpath('li')
        for li in lis:
            html = li.get_attribute('innerHTML')
            soup = BeautifulSoup(html, 'lxml')
            url = soup.a.img['data-original']
            name = soup.h3.string
            self.download(url, name)

    def main(self):
        page = 0
        self.parse()
        while True:
            if EC.element_to_be_clickable((By.XPATH, '//*[@id="Main"]/div[4]/div/span[12]/a')):
                page += 1
                print(f'正在下载第{page}页图片!')
                Next = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="Main"]/div[4]/div/span[12]/a'))).click()
                time.sleep(1)
                self.parse()
        self.driver.quit()


if __name__ == '__main__':
    a = spider()
    a.main()
