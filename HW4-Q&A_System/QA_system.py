import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

from cleanco import cleanco
import spacy 
import en_core_web_sm

import re

import datetime

import os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import sys

from whoosh.qparser import QueryParser
from whoosh import scoring
from whoosh.index import open_dir


nlp = en_core_web_sm.load()


def calc_cosine_similarity(X, Y):
    # tokenization 
    X_list = word_tokenize(X)  
    Y_list = word_tokenize(Y) 

    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 

    # remove stop words from the string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 

    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0

    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    return cosine

def get_question_type(question):
    template_1 = 'Which companies went bankrupt in month X of year Y'
    template_2 = 'What affects GDP'
    template_3 = 'What percentage of drop or increase is associated with Z'
    template_4 = 'Who is the CEO of company X'
    
    template_list = [template_1, template_2, template_3, template_4]
    
    max_cosine_similarity = 0
    question_type = -1
    for i in range(len(template_list)):
        template = template_list[i]
        cosine_similarity = calc_cosine_similarity(question, template)
#         print(cosine_similarity)
        if cosine_similarity > max_cosine_similarity:
            max_cosine_similarity = cosine_similarity
            question_type = i
    return question_type

    
def clean_company_name(company):
    clean_company = cleanco(company).clean_name()
    return clean_company

def get_bankrupt_from_date(year, month):
    
    
    all_month_list = ['January', 'February', 'March', 'April' , 'May', 'June', 'July' , 'August', 'September', 'October', 'November', 'December']
    all_year_list = [str(i) for i in range(1900,2014)]
    
    
    ix = open_dir("indexdir")
    key_word_list = ['bankrupt', 'bankruptcy', 'Chapter', 'liquidation', 'liquidate']
    # query_str is query string
    query_str = ' OR '.join(key_word_list)
    # Top 'n' documents as result
    topN = 460
    candidate_entities = {}
    recent_day_list = ['today', 'yesterday', 'this Monday', 'this Tuesday', 'this Wednesday', 'this Thursday', 'this Friday', 'this week' 'last week']

    key_phrase_list = ['went bankrupt', 'filed for bankruptcy', 'declared Chapter', 'liquidated', 'went liquidation']

    with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query,limit=topN)
#             print(results)
        for i in range(topN):
    #         print(results[i]['title'], str(results[i].score))
            raw_text = results[i]['textdata']
            raw_text = raw_text.replace('Mt. Gox', 'Mt.Gox')
            sentence_list = nltk.sent_tokenize(raw_text) 
            for j in range(len(sentence_list)):
                sentence = sentence_list[j]
                sentence = sentence.replace('Mt.Gox', 'Mt. Gox')
#                     key_word_match = False
#                     for key_word in key_word_list:
#                         if key_word in sentence:
#                             print('ddddddddddddddd', sentence)
                recent_date_match = False       
                for key_phrase in key_phrase_list:
                    if key_phrase in sentence:
                        key_phrase_pos = sentence.find(key_phrase)

                        file_date = datetime.datetime.strptime(results[i]['title'][:10],'%Y-%m-%d')
                        file_year = file_date.strftime('%Y')
                        file_month = file_date.strftime('%B')


                        have_specific_month = False
                        for temp_month in all_month_list:
                            if temp_month in sentence:
                                have_specific_month = True

                        have_specific_year = False
                        for temp_year in all_year_list:
                            if temp_year in sentence:
                                have_specific_year = True

                        if (month in sentence) and ((year in sentence) or ((year == file_year) and not have_specific_year)):


#                             print('Time Match', sentence )
                            nlp_result = nlp(sentence)
                            ents = nlp_result.ents
                            for ent in ents:
                                if ent.label_ == 'ORG' or ent.label_ == 'PERSON':
                                    if (ent.end_char == key_phrase_pos-1):
#                                         print(sentence)
                                        found_date = results[i]['title'][:10]
                                        cleaned_company_name = clean_company_name(ent.text)
                                        if cleaned_company_name not in candidate_entities.keys():
                                            candidate_entities[cleaned_company_name] = [found_date]
                                        else:
                                            candidate_entities[cleaned_company_name].append(found_date)



                        if not have_specific_month and not have_specific_year:
                            if (file_year == year) and (file_month == month):
#                                 print(file_year, year, file_month, month)
#                                 print('File Match', sentence)
#                                 for recent_day in recent_day_list:
#                                     if recent_day in sentence:
#                                         recent_date_match = True
#                                 if recent_date_match:
                                nlp_result = nlp(sentence)
                                ents = nlp_result.ents

                                phrase_found = False
                                for ent in ents:
                                    if ent.label_ == 'ORG' or ent.text=='Mt. Gox':
                                        if (ent.end_char == key_phrase_pos-1):
                                            phrase_found = True 
                                            found_date = results[i]['title'][:10]
                                            cleaned_company_name = clean_company_name(ent.text)
                                            if cleaned_company_name not in candidate_entities.keys():
                                                candidate_entities[cleaned_company_name] = [found_date]
                                            else:
                                                candidate_entities[cleaned_company_name].append(found_date)

#                                     if sentence[(key_phrase_pos-1-len('company')):(key_phrase_pos-1)] == 'company':
#                                         print('COMPANY', sentence_list[j-1])
#                                     if sentence[(key_phrase_pos-1-len('and')):(key_phrase_pos-1)] == 'and':
#                                         print('AND', sentence_list[j])


#         remove_str_list = ['cftc', 'reuters']
#         for remove_str in remove_str_list: 
#             if remove_str in candidate_entities:
#                 candidate_entities.remove(remove_str)
    return candidate_entities

def get_gdp_from_factor(factor):
    look_length_left = 80

    key_word = 'gdp'
    
    closest_distance = look_length_left


    excluded_word_list = ['up to', 'down to']
    up_word_list = ['up', 'increase', 'rise', 'add']
    down_word_list = ['down', 'decrease', 'drop']
    change_word_list = up_word_list + down_word_list


    ix = open_dir("indexdir")
    query_str = factor + ' AND ' + '(' + 'GDP' +')'
    topN = 100
    result = []
    with ix.searcher(weighting=scoring.TF_IDF()) as searcher2:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher2.search(query,limit=topN)
#         print(results)
        found = False
        direction = ''
        found_date = ''
        for i in range(topN):
#             print(results[i]['title'], str(results[i].score))
            raw_text = results[i]['textdata'].lower()
            sentence_list = nltk.sent_tokenize(raw_text)  
            for sentence in sentence_list:
                sentence_ignore = False
                for excluded_word in excluded_word_list:
                    if excluded_word in sentence:
                        sentence_ignore = True
    #             sentence_token_list = word_tokenize(sentence)
    #             sentence_token_tag_list = nltk.pos_tag(sentence_token_list)
                
                if not sentence_ignore:
                    key_word_match = False
                    if key_word.lower() in sentence:
                        key_word_match = True
                    
                    if key_word_match and (factor in sentence):
#                         print(sentence)
                        for up_word in up_word_list:
                            if up_word in sentence:
                                key_word_pos_list = [i.start() for i in re.finditer(key_word, sentence)]
                                for key_word_pos in key_word_pos_list:
                                    if key_word_pos >=3:
                                        if sentence[key_word_pos-3:key_word_pos-1] == 'of':
                                            break
                                    word_pos = sentence.find(up_word)
                                    start = max(0,word_pos-look_length_left)
                                    end = min(len(sentence)+1,word_pos+look_length_left+6)
#                                     print(start)
#                                     print(end)
                                    sentence_snippet = sentence[start:end]
                                    key_word_match = False
                                    if key_word.lower() in sentence_snippet:
                                        key_word_match = True
                                    if (key_word_pos > word_pos) and key_word_match and factor in sentence_snippet:      
                                        all_percent_list = extract_all_percent_re(sentence[word_pos:end])
                                        if len(all_percent_list) > 0:
        #                                     if (all_percent_list) == 1:
        #                                         closest_percent = all_percent_list[0]
        #                                     else:
                                            for percent in all_percent_list:
                                                percent_start = sentence.find(percent)
                                                if percent_start > word_pos:
                                                    distance = abs(percent_start+len(percent)-key_word_pos)
        #                                             else:
        #                                                 distance = abs(percent_start-word_pos)
                                                    if distance < closest_distance:
                                                        closest_percent = percent
                                                        closest_distance = distance
                                                        found = True
#                                                         print(sentence_snippet)
                                                        direction = 'up'
                                                        found_date = results[i]['title'][:10]
#                                             print(sentence_snippet)
#                                             print(all_percent_list)
        #                                     result = closest_percent
    #                     if found:
    #                         break

                        for down_word in down_word_list:
                            if down_word in sentence:
                                key_word_pos_list = [i.start() for i in re.finditer(key_word, sentence)]
                                for key_word_pos in key_word_pos_list:
                                    if key_word_pos >=3:
                                        if sentence[key_word_pos-3:key_word_pos-1] == 'of':
                                            break
                                    word_pos = sentence.find(down_word)
                                    start = max(0,word_pos-look_length_left)
                                    end = min(len(sentence)+1,word_pos+look_length_left+6)
#                                     print(start)
#                                     print(end)
                                    sentence_snippet = sentence[start:end]
                                    key_word_match = False
                                    if key_word.lower() in sentence_snippet:
                                        key_word_match = True
                                    if (key_word_pos > word_pos) and key_word_match and factor in sentence_snippet:      
                                        all_percent_list = extract_all_percent_re(sentence[word_pos:end])
                                        if len(all_percent_list) > 0:
        #                                     if (all_percent_list) == 1:
        #                                         closest_percent = all_percent_list[0]
        #                                     else:
                                            for percent in all_percent_list:
                                                percent_start = sentence.find(percent)
                                                if percent_start > word_pos:
                                                    distance = abs(percent_start+len(percent)-key_word_pos)
        #                                             else:
        #                                                 distance = abs(percent_start-word_pos)
                                                    if distance < closest_distance:
                                                        closest_percent = percent
                                                        closest_distance = distance
                                                        found = True
#                                                         print(sentence_snippet)
                                                        direction = 'down'
                                                        date = ''
                                                        found_date = results[i]['title'][:10]
#                                         print(sentence)
#                                         print(sentence_snippet)
#                                         print(all_percent_list)
    #                                     result = closest_percent
    #                                     break
#                     if found:
#                         break
#                 if found:
#                     break
#             if found:
#                 break
        if found:
            return closest_percent, direction, found_date
        else:
            return -1, 'not found', found_date



def get_ceo_from_company(company):
#     look_length_left = 80
    key_word_list = ['CEO', 'chief executive officer']
    
    
    closest_distance = 500


#     excluded_word_list = ['up to', 'down to']
#     up_word_list = ['up', 'increase', 'rise', 'add']
#     down_word_list = ['down', 'decrease', 'drop']
#     change_word_list = up_word_list + down_word_list


    ix = open_dir("indexdir")
    query_str = company + ' AND ' + '(' + 'GDP' +')'
    topN = 100
    result = []
    with ix.searcher(weighting=scoring.TF_IDF()) as searcher2:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher2.search(query,limit=topN)
#         print(results)
        found = False
        for i in range(topN):
            try: 
                results[i]
            except:
                break
                
#             print(results[i]['title'], str(results[i].score))
            raw_text = results[i]['textdata']
            sentence_list = nltk.sent_tokenize(raw_text)  
            for sentence in sentence_list:
                sentence_ignore = False
#                 for excluded_word in excluded_word_list:
#                     if excluded_word in sentence:
#                         sentence_ignore = True
    #             sentence_token_list = word_tokenize(sentence)
    #             sentence_token_tag_list = nltk.pos_tag(sentence_token_list)
                if not sentence_ignore:
                    sentence = sentence.replace('REUTERS/Kim White', '')
                    key_word_match = False
                    for key_word in key_word_list:
                        if key_word.lower() in sentence.lower():
                            key_word_match = True
                    if key_word_match and (company in sentence):
                        for key_word in key_word_list:
#                             print(sentence)
                            key_word_pos_list = [i.start() for i in re.finditer(key_word, sentence)]
                            for key_word_pos in key_word_pos_list:
    #                             if key_word_pos >=3:
    #                                 if sentence[key_word_pos-3:key_word_pos-1] == 'of':
    #                                     break
                                company_pos = sentence.find(company)
                                
                                ents = nlp(sentence).ents
                                for ent in ents:
                                    person_name_list = ent.text.split(' ')
                                    if len(person_name_list) >= 2:
                                        first_name = ent.text.split(' ')[0]
                                        last_name = ent.text.split(' ')[1]
                                        if (len(re.findall(r'[A-Z]',first_name)) == 1) and (len(re.findall(r'[A-Z]',last_name)) == 1):
                                            if (ent.label_ == 'PERSON') and ent.text:

                                                if ent.text.lower()[-2:] == '\'s': 
                                                    person_name = ent.text[:-2]
        #                                             name_has_possesive_at_end.append(True)
                                                else:
                                                    person_name = ent.text
        #                                             name_has_possesive_at_end.append(False)

                                                person_pos = ent.start_char

                                                if person_pos < key_word_pos:
                                                    distance = abs(person_pos+len(ent.text)-key_word_pos)
                                                else:
                                                    distance = abs(person_pos-(key_word_pos+len(key_word)))
                                                if distance < closest_distance:
                                                    closest_distance = distance
                                                    closest_person = ent.text
                                                    found = True
                                                    found_date = results[i]['title'][:10]
        if found:
            return closest_person, found_date
        else:
            return 'not found', 'not found'

def bankrupt_answer(question):
    nlp_result = nlp(question)
    ents = nlp_result.ents
    month = ''
    year = ''
    for ent in ents:
        # 391 label are dates entities
        if ent.label == 391:
            for tag in pos_tag(word_tokenize(ent.text)):
                if tag[1] == 'NNP':
                    month = tag[0]
                elif tag[1] == 'CD':
                    year = tag[0]
    if month == '' or year == '':
        print(f'Please specify both month and year.')
        return
    
    company_dict = get_bankrupt_from_date(year, month)
    if company_dict == {}:
        print(f'No company is found to declare banruptcy in {month} of {year}.')
    elif len(company_dict.keys()) == 1:
        print(f'{len(company_dict.keys())} company is found to declare banruptcy in {month} of {year}.')
    else:
        print(f'{len(company_dict.keys())} companies are found to declare banruptcy in {month} of {year}.')
        
    for key, value in company_dict.items():
        articles = ' '.join(value)
        print(f'{key}\'s bankruptcy is supported by articles from {articles}.')
    return

def gdp_factors(question):
    GDP_factor_list = ['consumption', 'government spending', 'investment', 'exports']
    answer = 'GDP is affected by ' + ', '.join(GDP_factor_list) + '.'
    print(answer)
    return

def extract_all_percent_re(article_str):
    percent_signs = re.findall('[0-9+-.,/]+%', article_str)
    percent_texts = re.findall('[0-9+-.,/a-zA-Z]* ?percent+[a-zA-Z]*', article_str)
    percent_point_texts = re.findall('[0-9+-.,/a-zA-Z]*\s?percent+[a-zA-Z]*\s?(?:point|Point)s?', article_str)
    of_a_percent_point_texts = re.findall('[0-9+-.,/a-zA-Z]*\sof\sa\s?percent+[a-zA-Z]*\s?(?:point|Point)s?', article_str)
    return percent_signs + percent_texts + percent_point_texts + of_a_percent_point_texts

def gdp_factor_change(question):
    GDP_factor_list = ['consumption', 'government spending', 'investment', 'exports']
    direction = 'factor not found'
    for GDP_factor in GDP_factor_list:
        if GDP_factor in question:
            percent, direction, found_date = get_gdp_from_factor(GDP_factor)
            input_factor = GDP_factor
            if percent != -1:
                if 'percentage' in percent.split(' ')[-1]:
                    percent = percent + ' points'
            break
    if direction == 'factor not found':
        print('The factor you asked is not identified as a factor that affects GDP')
    elif direction == 'not found':
        print('The current database does not have information about this factor\'s impact on GDP')
    elif direction == 'up':
        print(f'{input_factor.capitalize()} increased GDP by {percent} according to a {found_date} article.')
    elif direction == 'down':
        print(f'{input_factor.capitalize()} decreased GDP by {percent} according to a {found_date} article.')
    return
        
def get_company_ceo(question):
    nlp_result = nlp(question)
    ents = nlp_result.ents
    company_found = False
    for ent in ents:
        # 391 label are dates entities
        if ent.label_ == "ORG":
            company = ent.text
            company_found = True
            
    if not company_found:
        company_name_list = []
        for token_pos in pos_tag(word_tokenize(question))[1:]:
            if (token_pos[0].istitle() and (token_pos[1] == 'NNP')):
                company_name_list.append(token_pos[0])
        if len(company_name_list) == 0:        
            print('The company name asked is invalid or not specified, please specify the company name.')
            return
        else:
            company = ' '.join(company_name_list)
    ceo, found_date = get_ceo_from_company(company)
    if ceo == 'not found':
        print(f'The CEO of {company} is not found')
    else:
        print(f'The CEO of {company} is {ceo} according to a {found_date} article')
    
    return


def main(question):
    question_type = get_question_type(question)   
    if question_type == -1:
        print('Sorry we could not answer this question')
    else:
#         print(question_type)
        if question_type == 0:
            answer = bankrupt_answer(question)
        elif question_type == 1:
            answer = gdp_factors(question)
        elif question_type == 2:
            answer = gdp_factor_change(question)
        elif question_type == 3:
            answer = get_company_ceo(question)
    return