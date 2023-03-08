"""Download all IR spectra and structure files available from SDBS."""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from time import sleep
import re
import os
import shutil
import pyautogui


# Define paths.
other_path = '../sdbs_dataset/other/'
gif_path = '../sdbs_dataset/gif/'
ids_path = '../sdbs_dataset/sdbs_ids.txt'
down_path = '/Users/guwon/Downloads/' # Change path to your Downloads folder.

# Define URLs for get requests.
disclaimer = 'https://sdbs.db.aist.go.jp/sdbs/cgi-bin/cre_disclaimer.cgi?REQURL=/sdbs/cgi-bin/direct_frame_top.cgi&amp;REFURL='
main = 'https://sdbs.db.aist.go.jp/sdbs/cgi-bin/landingpage?sdbsno='

# Define regex strings used to match.
formula = re.compile('Molecular Formula: (.*?)$')
mw = re.compile('Molecular Weight: (.*?)$')
inchi = re.compile('InChI: (InChI=.*?)$')
inchikey = re.compile('InChIKey: (.*?)$')
cas = re.compile('RN: (.*?)$')
name = re.compile('Description: Compound Name: (.*?)$')


def check_dir():
    """Check if file directories exist and create them if they do not."""
    if not os.path.exists('../sdbs_dataset/'):
        os.makedirs('../sdbs_dataset')
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    if not os.path.exists(other_path):
        os.makedirs(other_path)
    if not os.path.exists(down_path):
        print('Please set the down_path on line 21 to your Downloads folder.')
    # Check if IDs file exists.
    if not os.path.exists(ids_path):
        print('sdbs_ids.txt does not exist')
    elif os.path.exists(ids_path):
        print('sdbs_ids.txt alrealdy exists')
        ids = [line.rstrip('\n') for line in open(ids_path)]
    return ids

# Check directories.
ids = check_dir()


# Install latest driver for Chrome.
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(disclaimer)
driver.find_element("xpath", '/html/body/center/form/table/tbody/tr[6]/td/input[4]').click()
driver.switch_to.window(driver.window_handles[0])


# Define variables.
errors = []
count = 0
# Loop through each ID in the list and download.
for j in ids[:]:
    # Attempt to download information for selected ID.
    try:
        count += 1
        print('\nSession download count: %s' % count)

        # After scraping 30 unique compounds, start a new browser.
        if count % 30 == 0:
            print('New disclaimer')
            driver.get(disclaimer)
            driver.find_element("xpath", '/html/body/center/form/table/tbody/tr[6]/td/input[4]').click()
            sleep(1)

        # Select first tab in the controlled window.
        driver.switch_to.window(driver.window_handles[0])
        # Open profile of selected compound.
        driver.get(main + str(j))
        sleep(1)

        # Create new file or skip if already exists.
        filepath = os.path.join(other_path, '%s_other.txt' % j)
        if not os.path.isfile(filepath):
            print('Downloading: %s_other.txt' % j)
            otherpath = os.path.join(other_path, '%s_other.txt' % j)
            file = open(otherpath, 'w+')
            # Grab texts with relevant information and write in file.
            texts = driver.find_elements(By.XPATH, '//tr')
            for txt in texts:
                text = txt.text.replace('\n', ': ')
                if re.search(formula, text):
                    file.write('Formula: %s' % re.search(formula, text).group(1))
                    file.write('\n')
                if re.search(mw, text):
                    file.write('Molecular weight: %s' % re.search(mw, text).group(1))
                    file.write('\n')
                if re.search(inchikey, text):
                    file.write('InChIKey: %s' % re.search(inchikey, text).group(1))
                    file.write('\n')
                if re.search(cas, text):
                    file.write('CAS: %s' % re.search(cas, text).group(1))
                    file.write('\n')
                if re.search(name, text):
                    file.write('Compound name: %s' % re.search(name, text).group(1).replace(':', ';'))
                    file.write('\n')
                if re.search(inchi, text):
                    file.write('InChI: %s' % re.search(inchi, text).group(1).replace(':', ';'))
                    file.write('\n')
            file.close()
        else:
            print('%s_other.txt already exists' % j)

        # Check if IR spectra exists.
        ir = []
        elems = driver.find_elements(By.XPATH, '//a')
        for elem in elems:
            if 'IR' in elem.text:
                ir.append(elem.text)
        # Loop through all available IR spectra and download.
        for i in ir:
            driver.get(main + str(j))
            method = re.search('IR : (.*) IR', i).group(1).split(' ')[0]
            print(method)
            picpath = os.path.join(gif_path, str(j) + '_' + method + '.gif')
            # Check if spectrum exists otherwise download.
            if not os.path.isfile(picpath):
                filename = str(j) + '_' + method + '.gif'
                print('Downloading: ' + filename)
                if 'CCl4' in method:
                    driver.find_element(By.XPATH, "//a[contains(text(), '" + "IR : CCl" + "')]").click()
                else:
                    driver.find_element(By.XPATH, "//a[contains(text(), '" + i + "')]").click()
                driver.find_element(By.XPATH, '//input[@value="I agree the disclaimer and use SDBS"]').click()
                sleep(7)
                pyautogui.rightClick(x=814, y=504)
                sleep(3)
                pyautogui.press('down')
                sleep(1)
                pyautogui.press('down')
                sleep(1)
                pyautogui.press('enter')
                sleep(4)
                pyautogui.write(filename)
                pyautogui.hotkey('enter')
                sleep(1)
                # Move downloaded spectrum from Downloads folder to gif folder.
                shutil.move(down_path + filename, picpath)
                print('File moved to gif folder')
            else:
                print(filename + '.gif already exists')
                continue
    except:
        errors.append(j)
        print('Error IDs this session: ', errors)
        continue