from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import json
def scrape_FAQ(url):
    def getting_content(href):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(href)
            page.wait_for_timeout(3000)
            PageContent = page.content()
            browser.close()
        return PageContent
    def extract_QA(FAQpage):
        FAQpage = BeautifulSoup(FAQpage, 'lxml')
        questions = []
        questionsDiv = FAQpage.find_all('div', {'class': 'question'})
        for i in range(1, len(questionsDiv)):
            clean_question = re.sub(r'[^\u0600-\u06FFa-zA-Z\s؟?]', '', questionsDiv[i].find('h4').text.strip())
            questions.append(clean_question)
        answers = []
        answersDiv = FAQpage.find_all('div', {'class': 'my-1'})
        for i in range(1, len(answersDiv)):
            clean_answer = answersDiv[i].get_text(separator=" ", strip=True)
            answers.append(clean_answer)
        QA = {}
        for i in range(len(questions)):
            QA[questions[i]] = answers[i]
        return(QA)


    mobileFAQpage=getting_content("https://te.eg/about-te/faq")
    landlineFAQpage=getting_content("https://te.eg/about-te/faq/fixed-voice")
    internetFAQpage=getting_content("https://te.eg/about-te/faq/fixed-broadband")
    all_faqs={"mobile": extract_QA(mobileFAQpage),"landline":extract_QA(landlineFAQpage),"internet":extract_QA(internetFAQpage)}
    with open("WE_FAQ_data.json", "w", encoding="utf-8") as json_file:
        json.dump(all_faqs, json_file, ensure_ascii=False, indent=4)
def scrape_Bracnhes():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.te.eg/store-locator")
        page.wait_for_timeout(3000)
        dropdown = page.locator('#governorate')
        dropdown.click()
        page.wait_for_selector('ng-dropdown-panel')
        options_locator = page.locator('ng-dropdown-panel .ng-option')
        all_governorates = options_locator.all_text_contents()
        branches={"governorates":{}}
        for i in range(len(all_governorates)):
            page.click('#governorate')
            page.wait_for_selector('ng-dropdown-panel')
            current_gov_option = page.locator('ng-dropdown-panel .ng-option').nth(i)
            current_gov_option.click()
            page.wait_for_timeout(1500)
            page.click('#city')
            page.wait_for_selector('ng-dropdown-panel')
            districts_locator = page.locator('ng-dropdown-panel .ng-option')
            all_districts = districts_locator.all_text_contents()
            for j in range(len(all_districts)):
                page.click('#city')
                page.wait_for_selector('ng-dropdown-panel')
                current_district_option = page.locator('ng-dropdown-panel .ng-option').nth(j)
                current_district_option.click()
                page.wait_for_timeout(3000)
                PageContent=page.content()
                PageContent=BeautifulSoup(PageContent,'lxml')
                DesiredCity=all_governorates[i]
                DesiredDistrict=all_districts[j]
                DesiredName=PageContent.find_all('p',{'class':'store-name'})
                DesiredAddress=PageContent.find_all('p',{'class':'store-address'})
                branchWorkTime="من السبت إلى الأربعاء من الساعة 9:00 صباحًا حتى 10:30 مساءً و الخميس من الساعة 9:00 صباحًا حتى 11:30 مساءً و الجمعة من الساعة 3:00 مساءً حتى 11:30 مساء"
                if DesiredCity not in branches["governorates"]:
                     branches["governorates"][DesiredCity] = {"districts": {}}
                if DesiredDistrict not in branches["governorates"][DesiredCity]["districts"]:
                    branches["governorates"][DesiredCity]["districts"][DesiredDistrict] = []
                for k in range(len(DesiredAddress)):
                    branchName=DesiredName[k].text.strip()
                    branchAddress=DesiredAddress[k].text.strip()
                    branches["governorates"][DesiredCity]["districts"][DesiredDistrict].append({"name":branchName,"address":branchAddress,"work_time":branchWorkTime})

            page.keyboard.press("Escape")

        browser.close()
    with open("WE_Branches_data.json", "w", encoding="utf-8") as json_file:
        json.dump(branches, json_file, ensure_ascii=False, indent=4)






scrape_FAQ('https://te.eg/about-te/faq')
scrape_Bracnhes()


