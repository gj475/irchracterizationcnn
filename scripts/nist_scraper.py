"""Download all IR spectra and structure files available from nist chemistry webbook."""

from bs4 import BeautifulSoup
import csv
import os
import requests
import re
import lxml.html


# Define paths.
jdx_path = '../nist_dataset/jdx/'
mol_path = '../nist_dataset/mol/'
sdf_path = '../nist_dataset/sdf/'
inchi_path = '../nist_dataset/inchi/'
inchikey_path = '../nist_dataset/inchikey/'
name_path = '../nist_dataset/name/'
mw_path = '../nist_dataset/mw/'

# Define url for get requests.
nist_url = 'http://webbook.nist.gov/cgi/cbook.cgi'

# Define regex for nist IDs.
id_re = re.compile('/cgi/cbook.cgi\?ID=(.*?)&')


def check_dir():
    """Check if file directories exist and create them if they do not."""
    if not os.path.exists('../nist_dataset/'):
        os.makedirs('../nist_dataset')
    if not os.path.exists('../nist_dataset/species.txt'):
        print('species.txt required in this directory for formula or name search.')
    if not os.path.exists(jdx_path):
        os.makedirs(jdx_path)
    if not os.path.exists(mol_path):
        os.makedirs(mol_path)
    if not os.path.exists(sdf_path):
        os.makedirs(sdf_path)
    if not os.path.exists(inchi_path):
        os.makedirs(inchi_path)
    if not os.path.exists(inchikey_path):
        os.makedirs(inchikey_path)
    if not os.path.exists(name_path):
        os.makedirs(name_path)
    if not os.path.exists(mw_path):
        os.makedirs(mw_path)


def get_jdx(nistid, indices):
    """Download ir jdx file for specified nist id, if not already downloaded."""
    for index in range(0, indices + 1):
        filepath = os.path.join(jdx_path, '%s_%s.jdx' % (nistid, index))
        if os.path.isfile(filepath):
            print('%s_%s.jdx: already exists' % (nistid, index))
            continue
        params = {'JCAMP': nistid, 'Type': 'IR', 'Index': index}    
        response = requests.get(nist_url, params=params)
        if index == 0 and response.text.splitlines()[0] == '##TITLE=Spectrum not found.':
            print('%s_%s.jdx: file not found' % (nistid, index))
            return
        elif index > 0 and response.text.splitlines()[0] == '##TITLE=Spectrum not found.':
            return
        print('%s_%s.jdx: downloading' % (nistid,index))
        with open(filepath, 'wb') as file:
            file.write(response.content)


def get_mol(nistid):
    """Download mol file for specified nist id, if not already downloaded."""
    filepath = os.path.join(mol_path, '%s.mol' % nistid)
    if os.path.isfile(filepath):
        print('%s.mol: already exists' % nistid)
        return
    params = {'Str2File': nistid}
    response = requests.get(nist_url, params=params)
    if response.text.splitlines()[0] == '':
        print('%s.mol: file not found' % nistid)
        return
    print('%s.mol: downloading' % nistid)
    with open(filepath, 'wb') as file:
        file.write(response.content)


def get_sdf(nistid):
    """Download sdf file for specified nist id, if not already downloaded."""
    filepath = os.path.join(sdf_path, '%s.sdf' % nistid)
    if os.path.isfile(filepath):
        print('%s.sdf: already exists' % nistid)
        return
    params = {'Str3File': nistid}
    response = requests.get(nist_url, params=params)
    if response.text == '':
        print('%s.sdf: file not found' % nistid)
        return
    print('%s.sdf: downloading' % nistid)
    with open(filepath, 'wb') as file:
        file.write(response.content)


def get_inchi(nistid):
    """Download inchi file for specified nist id, if not already downloaded."""
    filepath = os.path.join(inchi_path, '%s.inchi' % nistid)
    if os.path.isfile(filepath):
        print('%s.inchi: already exists' % nistid)
        return
    params = {'GetInChI': nistid}
    response = requests.get(nist_url, params=params)
    if response.text == '':
        print('%s.inchi: file not found' % nistid)
        return
    print('%s.inchi: downloading' % nistid)
    with open(filepath, 'wb') as file:
        file.write(response.content)


def get_inchikey(nistid):
    """Get inchikey for specified nist id and store in a file, if not already done so."""
    filepath = os.path.join(inchikey_path, '%s_inchikey.txt' % nistid)
    if os.path.isfile(filepath):
        print('%s_inchikey.txt: already exists' % nistid)
        return
    params = {'ID': nistid, 'Units': 'SI'}
    response = requests.get(nist_url, params=params, stream=True)
    response.raw.decode_content = True
    html = lxml.html.parse(response.raw)
    inchikey = html.xpath('/html/body/main/ul/li/span[@style = "font-family: monospace;"]/text()')
    if not inchikey:
        print('%s_inchikey.txt: info not found' % nistid)
        return
    print('%s_inchikey.txt: downloading' % nistid)
    with open(filepath, 'w') as file:
        file.write('InChIKey: %s' % inchikey[0])


def get_name(nistid):
    """Get all names for specified nist id and store in a file, if not already done so."""
    filepath = os.path.join(name_path, '%s_name.txt' % nistid)
    if os.path.isfile(filepath):
        print('%s_name.txt: already exists' % nistid)
        return
    params = {'ID': nistid, 'Units': 'SI'}
    response = requests.get(nist_url, params=params, stream=True)
    response.raw.decode_content = True
    html = lxml.html.parse(response.raw)
    name = html.xpath('/html/body/main/h1[@id = "Top"]/text()')   
    formula = html.xpath('/html/body/main/ul[1]/li[1]/strong/a/text() | /html/body/main/ul[1]/li[1]/text() | /html/body/main/ul[1]/li[1]/*/text()')
    other = html.xpath('//li[contains(.,"Other names:")]/text()')
    file = open(filepath, 'w+')
    file.write('ID: %s' % nistid)
    file.write('\n')
    if name:
        file.write('Name: %s' % name[0])
        file.write('\n')
    elif not name:
        print('%s_name.txt: name not found' % nistid)
    if 'Formula' in formula:
        file.write(''.join(formula))
        file.write('\n')
    elif 'Formula' not in formula:
        print('%s_name.txt: formula not found' % nistid)
    if other:
        file.write('Other names: %s' % (other[0].strip('\n')).replace('\n', ' '))
        file.close()
    elif not other:
        print('%s_name.txt: other names not found' % nistid)
    if not name and not other and 'Formula' not in formula:
        print('%s_name.txt: all name info not found' % nistid)
        return
    print('%s_name.txt: downloading' % nistid)


def get_mw(nistid):
    """Get molecular weight for specific nist id and store in a file, if not already done so."""
    filepath = os.path.join(mw_path, '%s_mw.txt' % nistid)
    if os.path.isfile(filepath):
        print('%s_mw.txt: already exists' % nistid)
        return   
    params = {'ID': nistid, 'Units': 'SI'}
    response = requests.get(nist_url, params=params, stream=True)
    response.raw.decode_content = True
    html = lxml.html.parse(response.raw)
    mw = html.xpath('//li[contains(.,"Molecular weight")]/text()')
    if not mw:
        print('%s_mw.txt: info not found' % nistid)
        return
    print('%s_mw.txt: downloading' % nistid)
    with open(filepath, 'w') as file:
        file.write('Molecular weight:%s' % mw[0])


def search_formula(formula):
    """Single nist search using the specified formula query and return the matching nist ids."""
    print('Searching formula: %s' % formula)
    params = {'Formula': formula, 'Units': 'SI', 'NoIon': 'on', 'cIR': 'on'}
    response = requests.get(nist_url, params=params)
    soup = BeautifulSoup(response.text)
    ids = list(set([re.match(id_re, link['href']).group(1) for link in soup('a', href=id_re)]))
    print('Result: %s' % ids)
    return ids


def search_mw(mw):
    """Single nist search using the specfied molecular weight query and returns the matching nist ids."""
    print('Searching molecular weight: %s' % mw)
    params = {'Value': mw, 'VType': 'MW', 'Units': 'SI', 'cIR': 'on', "Formula": ''}
    response = requests.get(nist_url, params=params)
    soup = BeautifulSoup(response.text, features='lxml')
    ids = list(set([re.match(id_re, link['href']).group(1) for link in soup('a', href=id_re)]))
    print('Result %s' % ids)
    return ids


def search_name(name):
    """Single nist search using the specified name query and return the matching nist ids."""
    print('Searching name: %s' % name)
    params = {'Name': name, 'Units': 'SI', 'cIR': 'on'}
    response = requests.get(nist_url, params=params)
    soup = BeautifulSoup(response.text)
    ids = list(set([re.match(id_re, link['href']).group(1) for link in soup('a', href=id_re)]))
    print('Result: %s' % ids)
    return ids


def get_nistid_formula_name(file):
    """Return all nist ids matched using formula or name query. Search name only if formula does not exist."""
    nistids = []
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[1]:
                nistids += list(set(search_formula(row[1])))
            elif not row[1]:
                nistids += list(set(search_name(row[0])))
    return nistids


def get_nistid_mw(num, current):
    """Return all nist ids matched using the molecular weight query."""
    if os.path.exists('../nist_dataset/ids.txt'):
        print('ids.txt already exists')
        nistids = []
        file = open('../nist_dataset/ids.txt')
        for i, line in enumerate(file):
            if i > current:
                nistids.append(line.rstrip('\n'))
        file.close()
        return nistids
    elif not os.path.exists('../nist_dataset/ids.txt'):
        nistids = []
        for mw in range(1, num):
            nistids += list(set(search_mw(mw)))
            nistids = list(set(nistids))
        print('Total IDs: %s' % len(nistids))
        file = open('../nist_dataset/ids.txt', 'w+')
        for nistid in nistids:
            file.write(nistid)
            file.write('\n')
        file.close()
        return nistids


if __name__ == '__main__':
    """Search nist for all compounds with ir spectra and downloads jdx, mol, sdf, inchi, inchikey, name, molecular weight files."""
    # Molecular weight to start search from.
    num = 1

    # Differenciates multiple entries of same molecule.
    indices = 10

    # Position of the ID to start search from.
    current = 1000

    check_dir()

    nistids = get_nistid_mw(num, current)

    print('Downloading files')

    for nistid in nistids:
        print('ID count: %s' % (current - 1))
        get_jdx(nistid, indices)
        get_mol(nistid)
        get_sdf(nistid)
        get_inchi(nistid)
        get_inchikey(nistid)
        get_name(nistid)
        get_mw(nistid)
        current += 1

    print('Download complete')