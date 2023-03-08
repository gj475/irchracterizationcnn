"""Process all IR spectra from NIST and SDBS to CNN input format."""

from jcamp import JCAMP_reader
from os import listdir
from scipy import interpolate
from rdkit import Chem
from smarts import fg_list_original, fg_list_extended
from PIL import Image
import os
import numpy as np
import sys
import csv
import cv2
import re


# Set print options.
np.set_printoptions(threshold=sys.maxsize)

# Define paths.
nist_inchi_path = '../nist_dataset/inchi/'
nist_jdx_path = '../nist_dataset/jdx/'
sdbs_gif_path = '../sdbs_dataset/gif/'
sdbs_png_path = '../sdbs_dataset/png/'
sdbs_other_path = '../sdbs_dataset/other/'
save_path = '../processed_dataset/'


def convert_x(x_in, unit_from, unit_to):
    """Convert between micrometer and wavenumber."""
    if unit_to == 'micrometers' and x_out == 'MICROMETERS':
        x_out = x_in
        return x_out
    elif unit_to == 'cm-1' and unit_from in ['1/CM', 'cm-1', '1/cm', 'Wavenumbers (cm-1)']:
        x_out = x_in
        return x_out
    elif unit_to == 'micrometers' and unit_from in ['1/CM', 'cm-1', '1/cm', 'Wavenumbers (cm-1)']:
        x_out = np.array([10 ** 4 / i for i in x_in])
        return x_out
    elif unit_to == 'cm-1' and unit_from == 'MICROMETERS':
        x_out = np.array([10 ** 4 / i for i in x_in])
        return x_out


def convert_y(y_in, unit_from, unit_to):
    """Convert between absorbance and trasmittance."""
    if unit_to == 'transmittance' and unit_from in ['% Transmission', 'TRANSMITTANCE', 'Transmittance']:
        y_out = y_in
        return y_out
    elif unit_to == 'absorbance' and unit_from == 'ABSORBANCE':
        y_out = y_in
        return y_out
    elif unit_to == 'transmittance' and unit_from == 'ABSORBANCE':
        y_out = np.array([1 / 10 ** j for j in y_in])
        return y_out
    elif unit_to == 'absorbance' and unit_from in ['% Transmission', 'TRANSMITTANCE', 'Transmittance']:
        y_out = np.array([np.log10(1 / j) for j in y_in])
        return y_out


def get_png():
    """Convert GIF to PNG."""
    # Check if PNG folder exists.
    if not os.path.exists(sdbs_png_path):
        os.makedirs(sdbs_png_path)
    files = listdir(sdbs_gif_path)
    for file in files:
        if not file.startswith('.'):
            from_path = sdbs_gif_path + file
            img = Image.open(from_path)
            file_name = os.path.splitext(file)[0]
            to_path = sdbs_png_path + file_name + '.png'
            img.save(to_path, 'png')


def get_unique(x_in, y_in):
    """Removes duplicates in x and takes smallest y value for each x value."""
    x_out = sorted(list(set(x_in)), reverse=True)
    y_out = []
    for i in x_out:
        y_temp = []
        for ii, j in zip(x_in, y_in):
            if i == ii:
                y_temp.append(j)
        y_out.append(min(y_temp))
    return x_out, y_out


def get_contours(image):
    """Returns normalized coordinates of a spectrum."""
    # image = cv2.imread(sdbs_png_path + '/' + file, 0)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    # Define kernel length.
    kernel_length = np.array(image).shape[1] // 80
    # Verticle kernel of (1 * kernel_length) used to detect verticle lines in the image.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # Horizontal kernel of (kernel_length * 1) used to detect horizontal lines in the image.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # Kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Morphological operation to detect vertical lines from an image.
    vertical_temp = cv2.erode(thresh, vertical_kernel, iterations=3)
    verticle_lines = cv2.dilate(vertical_temp, vertical_kernel, iterations=3)
    # Morphological operation to detect horizontal lines from an image.
    horizontal_temp = cv2.erode(thresh, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(horizontal_temp, horizontal_kernel, iterations=3)
    # Add two images with specific weight parameters to get a third summation image.
    image = cv2.addWeighted(verticle_lines, 0.5, horizontal_lines, 0.5, 0.0)
    image = cv2.erode(~image, kernel, iterations=2)
    ret, thresh = cv2.threshold(image, 127,255, cv2.THRESH_BINARY)
    # Find contours in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_sdbs(fg_list):
    """Create dataset in CSV format."""
    # Process data from SDBS database.
    print('Start SDBS processing')
    for file in listdir(sdbs_png_path):
        try:
            if 'KBr' in file or 'liquid' in file or 'nujol' in file:
                if not file.startswith('.'):
                    original_image = cv2.imread(sdbs_png_path + '/' + file, 0)
                    # image = Image.open(sdbs_png_path + '/' + file)
                    height, width = original_image.shape
                    if width == 715:
                        sdbs_id = file.split('_')[0]
                        other_file = open(sdbs_other_path + '/' + sdbs_id + '_other.txt').readlines()
                        for line in other_file:
                            match = re.match('InChI: (.*)', line)
                            if match is not None:
                                print(file)
                                inchi = match.groups()[0]
                                inchi = inchi.split(':')[0]
                                mol = Chem.MolFromInchi(inchi)
                                contours = get_contours(original_image)
                                spectrum = []
                                ratio = 684 / 26
                                x_step_1 = (4000 - 2000) / (ratio * 10)
                                x_step_2 = (2000 - 400) / (ratio * 16)
                                y_step = 100 / 320
                                for contour in contours:
                                    # Returns location, width and height for every contour.
                                    x, y, w, h = cv2.boundingRect(contour)
                                    # Filter ROI.
                                    if 650 < w < 700 and 300 < h < 340:
                                        image = original_image[y - 2 : y + h + 2, x - 2 : x + w + 2]
                                        graph = image.shape
                                        for i in range(0, graph[1]):
                                            for j in range(0, graph[0]):
                                                if image[j, i] == 0:
                                                    spectrum.append([i, j])
                                x = []
                                y = []
                                for i in spectrum:
                                    if i[0] < ratio * 10:
                                        x.append(4000 - x_step_1 * i[0])
                                    elif ratio * 10 <= i[0]:
                                        x.append(2000 - x_step_2 * (i[0] - 262))
                                    y.append((100 - y_step * i[1]) / 100)
                                spectrum = get_unique(x, y)
                                x = np.linspace(4000, 400, 600)
                                f = interpolate.interp1d(spectrum[0], spectrum[1], kind='slinear')
                                y = f(x)
                                label_temp = []
                                if mol is not None:
                                    for fg in fg_list:
                                        pattern = Chem.MolFromSmarts(fg)
                                        match = mol.HasSubstructMatch(pattern)
                                        if match == True:
                                            label_temp.append(1)
                                        elif match == False:
                                            label_temp.append(0)
                                else:
                                    print('Error')
                                    continue
                                label_temp = np.append(label_temp, inchi)
                                label_temp = np.append(label_temp, 'sdbs')
                                x = np.append(x, inchi)
                                x = np.append(x, 1)
                                y = np.append(y, inchi)
                                y = np.append(y, 1)
                                with open(save_path + '/label_dataset.csv', mode='a') as label_data:
                                    y_data_writer = csv.writer(label_data, delimiter=',')
                                    y_data_writer.writerow(label_temp)
                                with open(save_path + '/input_dataset.csv', mode='a') as input_data:
                                    x_data_writer = csv.writer(input_data, delimiter=',')
                                    x_data_writer.writerow(y)
        except:
            print('Error')


def get_nist(fg_list):
    """"""
    # Process data from NIST database.
    print('Start NIST processing')
    for file in listdir(nist_jdx_path):
        try:
            nist_id = file.split('_')[0]
            inchi_file = nist_inchi_path + nist_id + '.inchi'
            if os.path.exists(inchi_file) == True:
                try:
                    jcamp_dict = JCAMP_reader(nist_jdx_path + file)
                except:
                    continue
                if jcamp_dict['x'] is None or len(jcamp_dict['x']) == 0:
                    continue
                if jcamp_dict['yunits'] is not None:
                    if jcamp_dict['yunits'] in ['dispersion index', 'absorption index', '(micromol/mol)-1m-1 (base 10)']:
                        continue   
                elif jcamp_dict['ylabel'] is not None:
                    if jcamp_dict['ylabel'] in ['dispersion index', 'absorption index', '(micromol/mol)-1m-1 (base 10)']:
                        continue
                if 'xunits' in jcamp_dict:
                    xunit = jcamp_dict['xunits']
                if 'yunits' in jcamp_dict:
                    yunit = jcamp_dict['yunits']
                if 'xlabel' in jcamp_dict:
                    xunit = jcamp_dict['xlabel']
                if 'ylabel' in jcamp_dict:
                    yunit = jcamp_dict['ylabel']
                x = jcamp_dict['x']
                y = jcamp_dict['y']
                x = convert_x(x, xunit, 'cm-1')
                y = convert_y(y, yunit, 'transmittance')
                y_min = min(y)
                y_max = max(y)
                x_min = min(x)
                x_max = max(x)
                if y_max > 1:
                    y = [1 if j > 1 else j for j in y]
                if y_min < 0:
                    y = [0 if j < 0 else j for j in y]
                if x[0] > x[1]:
                    x = x[::-1]
                    y = y[::-1]
                if x_max > 4000:
                    idx = next(i for i, xx in enumerate(x) if xx >= 4000) 
                    x = x[:idx + 1]
                    y = y[:idx + 1]
                if x_min < 400:
                    idx = next(i for i, xx in enumerate(x) if xx >= 400)
                    x = x[idx - 1:]
                    y = y[idx - 1:]
                if x_max < 4000:
                    x = np.append(x, [x[-1] + 1, 4000])
                    y = np.append(y, [y[-1], y[-1]])
                if x_min > 400:
                    x = np.insert(x, 0, x[0] - 1)
                    x = np.insert(x, 0, 400)
                    y = np.insert(y, 0, y[0])
                    y = np.insert(y, 0, y[0])
                print(file)
                inchi  = open(inchi_file).read()
                mol = Chem.MolFromInchi(inchi)
                spectrum = get_unique(x, y)
                x = np.linspace(4000, 400, 600)
                f = interpolate.interp1d(spectrum[0], spectrum[1], kind='slinear')
                y = f(x)
                label_temp = []
                if mol is not None:
                    for fg in fg_list:
                        pattern = Chem.MolFromSmarts(fg)
                        match = mol.HasSubstructMatch(pattern)
                        if match == True:
                            label_temp.append(1)
                        elif match == False:
                            label_temp.append(0)
                else:
                    print('Error')
                    continue
                label_temp = np.append(label_temp, inchi)
                label_temp = np.append(label_temp, 'nist')
                x = np.append(x, inchi)
                x = np.append(x, 2)
                y = np.append(y, inchi)
                y = np.append(y, 2)
                with open(save_path + '/label_dataset.csv', mode='a') as label_data:
                    y_data_writer = csv.writer(label_data, delimiter=',')
                    y_data_writer.writerow(label_temp)
                with open(save_path + '/input_dataset.csv', mode='a') as input_data:
                    x_data_writer = csv.writer(input_data, delimiter=',')
                    x_data_writer.writerow(y)
        except:
            print('Error')


if __name__ == '__main__':
    get_png() 
    # Selected either original or extended list.
    get_sdbs(fg_list_extended)
    get_nist(fg_list_extended)