##################
# input: predictions: {category: confidence, attributes: {value: confidence}}
# output: gender, list of search words for upperbody lowerbody and others
##################
##################
# issues:
# 1.
# garments parts division among upper+full, lower and commons is not perfect.
# but the frequency of wrong prediction would be very low.
# example: ruffle is currently considered upper+full body part. But there can be cases where lower apperal has ruffle.
#
# 2. confidence based picking of garments
# we are picking single garment for every supercategory (upper+full and lower) based on confidence.
# all the garment parts present are attached to the picked garment in supercategory. This cud be wrong prediction if the detected part is for apperal with lower confidence.
##################

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'FaceLib'))

from facelib import FaceDetector, AgeGenderEstimator
import cv2

sample_prediction = {
                        'pants': {
                                    'confidence': 67, 'attributes': {
                                                        'length': 'full',
                                                        'nickname': 'cargo (pants)'
                                                    }
                        },
                        'top, t-shirt, sweatshirt': {
                                                        'confidence': 89, 'attributes': {
                                                                            'nickname': 'classic (t-shirt)',
                                                                            'length': 'hip'
                                                                }
                        },
                        'sleeve': {
                                    'confidence': 45, 'attributes': {
                                                        'length': 'half',
                                                        'nickname': 'dropped-shoulder sleeve'
                                                    }

                        }
                    }

def divide_garment_parts():
    upperbody_parts = {'hood', 'collar', 'lapel', 'sleeve', 'neckline', 'epaulette', 'bow', 'bead', 'ruffle', 'fringe', 'ribbon', 'flower', 'rivet', 'sequin', 'tassel'} #bow, bead, zipper, ruffle, fringe, ribbon, flower, rivet, sequin, tassel
    lowerbody_parts = {'buckle'} #buckle
    common_parts = {'pocket'}

    return upperbody_parts, lowerbody_parts, common_parts

def get_gender(image_path):
    face_detector = FaceDetector()
    age_gender_detector = AgeGenderEstimator()

    image = cv2.imread(image_path)
    faces, boxes, scores, landmarks = face_detector.detect_align(image)
    if faces.tolist():
        genders, ages = age_gender_detector.detect(faces)
    else:
        genders = [None]
    # print(genders, ages)

    return genders[0]

def is_valid_nickname(nickname):
    valid_nicknames = ['classic (t-shirt)', 'polo (shirt)', 'undershirt', 'henley (shirt)', 'ringer (t-shirt)', 'raglan (t-shirt)', 'rugby (shirt)', 'sailor (shirt)', 'crop (top)', 'halter (top)', 'camisole', 'tank (top)', 'peasant (top)', 'tube (top)', 'tunic (top)', 'smock (top)', 'hoodie', 'blazer', 'pea (jacket)', 'puffer (jacket)', 'biker (jacket)', 'trucker (jacket)', 'bomber (jacket)', 'anorak', 'safari (jacket)', 'mao (jacket)', 'nehru (jacket)', 'norfolk (jacket)', 'classic military (jacket)', 'track (jacket)', 'windbreaker', 'chanel (jacket)', 'bolero', 'tuxedo (jacket)', 'varsity (jacket)', 'crop (jacket)', 'jeans', 'sweatpants', 'leggings', 'hip-huggers (pants)', 'cargo (pants)', 'culottes', 'capri (pants)', 'harem (pants)', 'sailor (pants)', 'jodhpur', 'peg (pants)', 'camo (pants)', 'track (pants)', 'crop (pants)', 'short (shorts)', 'booty (shorts)', 'bermuda (shorts)', 'cargo (shorts)', 'trunks', 'boardshorts', 'skort', 'roll-up (shorts)', 'tie-up (shorts)', 'culotte (shorts)', 'lounge (shorts)', 'bloomers', 'tutu (skirt)', 'kilt', 'wrap (skirt)', 'skater (skirt)', 'cargo (skirt)', 'hobble (skirt)', 'sheath (skirt)', 'ball gown (skirt)', 'gypsy (skirt)', 'rah-rah (skirt)', 'prairie (skirt)', 'flamenco (skirt)', 'accordion (skirt)', 'sarong (skirt)', 'tulip (skirt)', 'dirndl (skirt)', 'godet (skirt)', 'blanket (coat)', 'parka', 'trench (coat)', 'pea (coat)', 'shearling (coat)', 'teddy bear (coat)', 'puffer (coat)', 'duster (coat)', 'raincoat', 'kimono', 'robe', 'dress (coat )', 'duffle (coat)', 'wrap (coat)', 'military (coat)', 'swing (coat)', 'halter (dress)', 'wrap (dress)', 'chemise (dress)', 'slip (dress)', 'cheongsams', 'jumper (dress)', 'shift (dress)', 'sheath (dress)', 'shirt (dress)', 'sundress', 'kaftan', 'bodycon (dress)', 'nightgown', 'gown', 'sweater (dress)', 'tea (dress)', 'blouson (dress)', 'tunic (dress)', 'skater (dress)']
    if nickname in valid_nicknames:
        return True
    return False

def get_query_tags(upperbody_fullbody_garments, lowerbody_garments, others, garment_parts, upperbody_parts, lowerbody_parts, common_parts):
    #looking for nickname
    upperbody_query_words = set()
    lowerbody_query_words = set()
    other_query_words = set()

    for garment in upperbody_fullbody_garments:
        if 'nickname' in upperbody_fullbody_garments[garment]['attributes'].keys() and is_valid_nickname(upperbody_fullbody_garments[garment]['attributes']['nickname']['name']):
            upperbody_query_words.add(upperbody_fullbody_garments[garment]['attributes']['nickname']['name'])
        else:
            upperbody_query_words.add(garment)

        for value in upperbody_fullbody_garments[garment]['attributes'].keys():
            upperbody_query_words.add(upperbody_fullbody_garments[garment]['attributes'][value]['name'])

    for garment in lowerbody_garments:
        if 'nickname' in lowerbody_garments[garment]['attributes'].keys() and is_valid_nickname(lowerbody_garments[garment]['attributes']['nickname']['name']):
            lowerbody_query_words.add(lowerbody_garments[garment]['attributes']['nickname']['name'])
        else:
            lowerbody_query_words.add(garment)

        for value in lowerbody_garments[garment]['attributes'].keys():
            lowerbody_query_words.add(lowerbody_garments[garment]['attributes'][value]['name'])

    for garment in others:
        for value in others[garment]['attributes'].keys():
            other_query_words.add(others[garment]['attributes'][value]['name'])

    for garment_part in garment_parts:
        if garment_part in upperbody_parts:
            for value in garment_parts[garment_part]['attributes'].keys():
                upperbody_query_words.add(garment_parts[garment_part]['attributes'][value]['name']) #adding garment parts nickname e.g v-neck
        if garment_part in lowerbody_parts:
            for value in garment_parts[garment_part]['attributes'].keys():
                lowerbody_query_words.add(garment_parts[garment_part]['attributes'][value]['name'])
        if garment_part in common_parts:
            #TODO skipping for this version
            pass


    return upperbody_query_words, lowerbody_query_words, other_query_words

def pick_single_apperal_per_category(predictions):
    upperbody_garment_categories = {'top, t-shirt, sweatshirt', 'shirt, blouse', 'sweater', 'cardigan', 'jacket', 'vest'}
    lowerbody_garment_categories = {'pants', 'shorts', 'skirt', 'tights, stockings', 'leg warmer'}
    fullbody_garment_categories = {'coat', 'dress', 'jumpsuit'} #cape
    other_categories = {'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella'}
    garment_parts_categories = {'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel'}

    upperbody_fullbody_garments, lowerbody_garments, others, garment_parts = {}, {}, {}, {}
    lowerbody_max_confidence, upperbody_fullbody_max_confidence = 0, 0

    for prediction in predictions:
        if prediction in upperbody_garment_categories or prediction in fullbody_garment_categories:
            if predictions[prediction]['confidence'] > upperbody_fullbody_max_confidence:
                upperbody_fullbody_garments.clear()
                upperbody_fullbody_garments.update({prediction: predictions[prediction]})
                upperbody_fullbody_max_confidence = predictions[prediction]['confidence']
        elif prediction in lowerbody_garment_categories:
            if predictions[prediction]['confidence'] > lowerbody_max_confidence:
                lowerbody_garments.clear()
                lowerbody_garments.update({prediction: predictions[prediction]})
                lowerbody_max_confidence = predictions[prediction]['confidence']
        elif prediction in other_categories:
            others.update({prediction: predictions[prediction]})
        elif prediction in garment_parts_categories:
            garment_parts.update({prediction: predictions[prediction]})
        else:
            print('predicted category unknown!')

    return upperbody_fullbody_garments, lowerbody_garments, others, garment_parts

def fix_discrepency_cat_name(gender, upperbody_query_words, lowerbody_query_words, other_query_words):
    if 'shirt/blouse' in upperbody_query_words:
        upperbody_query_words.remove('shirt/blouse')
        upperbody_query_words.add('shirt')

    if 'top, t-shirt, sweatshirt' in upperbody_query_words:
        upperbody_query_words.remove('top, t-shirt, sweatshirt')
        if gender == 'Female':
            upperbody_query_words.add('top')
        else:
            upperbody_query_words.add('t-shirt')
        
    if 'Headband / Head Covering / Hair Accessory' in other_query_words:
        other_query_words.remove('Headband / Head Covering / Hair Accessory')
        other_query_words.add('Head accessory')

    if 'Bag/wallet' in other_query_words:
        other_query_words.remove('Bag/wallet')

    if 'Tight/Stockings' in lowerbody_query_words:
        lowerbody_query_words.remove('Tight/Stockings')
        lowerbody_query_words.add('tight')
        lowerbody_query_words.add('stockings')
    
    tags_to_remove = ['no non-textile material', 'no special manufacturing technique', 'no waistline']
    for tag in tags_to_remove:
        if tag in upperbody_query_words:
            upperbody_query_words.remove(tag)
        if tag in lowerbody_query_words:
            lowerbody_query_words.remove(tag)
        if tag in other_query_words:
            other_query_words.remove(tag)

    return upperbody_query_words, lowerbody_query_words, other_query_words

def map_fashionpedia2flipkart(image_path, predictions):

    upperbody_parts, lowerbody_parts, common_parts = divide_garment_parts() #for example sleeves goes into upper or full body garment parts and not lowerbody

    upperbody_fullbody_garments, lowerbody_garments, others, garment_parts = pick_single_apperal_per_category(predictions) #picking one with highest confidence

    upperbody_query_words, lowerbody_query_words, other_query_words = get_query_tags(upperbody_fullbody_garments, lowerbody_garments, others, garment_parts, upperbody_parts, lowerbody_parts, common_parts) # convert processed fashiopnedia predictions to flipkart searchable tags

    gender = get_gender(image_path)
    
    upperbody_query_words, lowerbody_query_words, other_query_words = fix_discrepency_cat_name(gender, upperbody_query_words, lowerbody_query_words, other_query_words)

    upperbody_index = -1
    lowerbody_index = -1
    for apparel in upperbody_fullbody_garments:
        upperbody_index = upperbody_fullbody_garments[apparel]['index']
    for apparel in lowerbody_garments:
        lowerbody_index = lowerbody_garments[apparel]['index']
    
    return gender, upperbody_query_words, lowerbody_query_words, other_query_words, upperbody_index, lowerbody_index
