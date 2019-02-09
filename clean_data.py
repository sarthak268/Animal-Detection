# -*-coding: utf-8 -*-

import json
from pascal_voc_writer import *
import shutil
import numpy as np
import csv
import random

def getName(id):
	if (id == 6):
		name = 'bobcat'
	elif (id == 1):
		name = 'opossum'
	#elif (id == 30):
	#	name = 'empty'
	elif (id == 9):
		name = 'coyote'
	elif (id == 3):
		name = 'raccoon'
	elif (id == 11):
		name = 'bird'
	elif (id == 8):
		name = 'dog'
	elif (id == 16):
		name = 'cat'
	elif (id == 5):
		name = 'squirrel'
	elif (id == 10):
		name = 'rabbit'
	elif (id  == 7):
		name = 'skunk'
	elif (id == 99):
		name = 'rodent'
	elif (id == 21):
		name = 'badger'
	elif (id == 34):
		name = 'deer'
	elif (id == 33):
		name = 'car'
	elif (id == 51):
		name = 'fox'
	return name

all_classes = [6, 1, 9, 3 , 11, 8, 16, 5, 10, 7, 99, 21, 34, 33, 51]
count = 0
arr = []
animal_count = {'bobcat' : 0, 'opossum' : 0, 'coyote': 0, 'raccoon' : 0, 'bird': 0, 'dog' : 0, 'cat' : 0, 'squirrel' : 0, 'rabbit' : 0, 'skunk': 0, 'rodent' : 0, 'badger': 0, 'deer' : 0, 'car' : 0, 'fox' : 0}

with open ('annotations.json', 'r') as f:
	json_data = json.load(f)
	for k in json_data['annotations']:
		print(count)
		if ('bbox' in k.keys()):
			count+=1
			bbox = k['bbox']
			xmin = ((int)(round(bbox[0])))
			xmax = (int)(round(bbox[0] + bbox[2]))
			ymin = (int)(round(bbox[1]))
			ymax = (int)(round(bbox[1] + bbox[3]))
			catergory_id = k['category_id']
			image_id = k['image_id']
			catergory_name = getName(catergory_id)
			animal_count[catergory_name] += 1
			data = ['./Test_Images_Final/' + (str)(image_id) + '.jpg', str(xmin), str(ymin), str(xmax), str(ymax), catergory_name]
			if (image_id + '.jpg' in os.listdir('./Test_Images/')):
				shutil.copy('./Test_Images/' + image_id + '.jpg', './Test_Images_Final/' + image_id + '.jpg')
				arr.append(data)

print (animal_count)

with open('annotations_test.csv', 'w+') as f:
	writer = csv.writer(f)
	writer.writerows(arr)
f.close()

print ('Total images', count)

count  = 0

class_ids = []
for i in range(len (all_classes)):
	id1 = all_classes[i]
	name = getName(id1)
	kuch = [name, str(i)]
	class_ids.append(kuch)
	count += 1
	
with open('classname2id.csv', 'w+') as writefile:
	writer = csv.writer(writefile)
	writer.writerows(class_ids)
writefile.close()

print (count)


