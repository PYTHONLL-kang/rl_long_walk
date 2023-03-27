from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time as t

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])

url_path = "https://vidkidz.tistory.com/2825"
driver = webdriver.Chrome(options=options)
driver.implicitly_wait(1)

driver.get(url_path) # url로 이동
driver.implicitly_wait(1) # wait time

try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="canvas"]'))
    )

finally:
    pass

for i in range(4):
    t.sleep(1)
    element.send_keys(Keys.SPACE)

print("space")

for st in range(1000):
    element.send_keys(Keys.ARROW_LEFT)
    t.sleep(0.001)
    print(st)
