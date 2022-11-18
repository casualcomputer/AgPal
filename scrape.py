# selenium 3
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
from numpy import random

driver=webdriver.Chrome() 

before_url = "https://agpal.ca/en/search-agpal?page="
after_url = "&pageSize=100&sort=title:asc"
agpal_url = [before_url+str(x)+after_url for x in range(1,32)]
output = []
html_source_codes = []

page = 0 #start the scrape from page 1

for url in agpal_url:
    page += 1  
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html)
    
    #save the html of each page
    html_source_codes.append(str(soup))
    
    time.sleep(3+min(15,abs(random.normal(loc=5, scale=3, size=1))[0]))
    
    counter = 0 # find html of the "article" section of each page
    for tag in soup.find_all("article"): # blank for many pages (varies during different runs)
        counter +=1  
        print("curr counter: ", counter)
        programTitle = [x.text for x in tag.find_all(attrs="h4 mrgn-tp-0 program__title")]
        infoCat = [x.text for x in tag.find_all(attrs="guides-information program__category__icon")]
        region =  [x.text for x in tag.find_all(attrs="across-canada program__regions__icon")]
        url = [x.find('a')['href'] for x in tag.find_all(attrs="url__value")]
        description = [x.text for x in tag.find_all(attrs="description__value")]
        organization = [x.text for x in tag.find_all(attrs="Organization")]
        contact = [x.text for x in tag.find_all(attrs="Contact")]
        output.append([programTitle,infoCat,region,url,description,organization,contact, page])
        
        print("--------------\n\n")
        
    print('finished page '+str(page)+"\n\n--------------\n\n--------------\n\n")
    
#part 1: export the output as a .csv file
out_data = pd.DataFrame(output)
out_data.columns = ["programTitle","infoCat","region","url","description","organization","contact","page"]
out_data.to_csv("agpal_search_results_run2.csv")
    #problem: I can't get the html of the <article> tag on my of the agpal pages?

#part 2: export source code of each search result page as .txt. files

#step 1: create empty .txt files
#file_name="" 
#for iterator in range(1,32,1): 
#    file_name="C:/Users/henry/OneDrive/Documents/Python Projects/AgPal/html_codes/"+"source_code_"+str(iterator)+".txt"    
#    with open(file_name,'w') as fp: 
#        index = iterator-1
#        fp.write(html_source_codes[index])
#    print('Done '+str(iterator)+'ith file !')

#to-do inject header and rotate ip: https://stackoverflow.com/questions/15645093/setting-request-headers-in-selenium