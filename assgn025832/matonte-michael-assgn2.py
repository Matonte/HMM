from collections import defaultdict
from itertools import product
import numpy as np
from math import log

DEBUG = False  # True to get printouts

def print_tagged_array(a, factor, pos_tag_list):
    print ('  valus *', factor)
    print ('START ', end='')
    for tag in pos_tag_list[1:]:
        print('%7s' % tag, end='')
    print()
    rows, cols = a.shape
    for row in range(rows):
        tag = pos_tag_list[row]
        print ('%5s ' % tag, end='')
        for col in range(cols):
            p = factor * a[row,col]
            print ('%7.2f' % p, end='')
        print()
    print() 
    
def print_array(a, factor):
    print ('  valus *', factor)
    rows, cols = a.shape
    for row in range(rows):
        for col in range(cols):
            p = factor * a[row,col]
            print ('%6.2f' % p, end='')
        print()
    print() 


def get_sentence_info(sentence_lines):
    for line in sentence_lines:
        words = [None]
        pos_list = ['START']
        for line in sentence_lines:
            if not line:
                continue
            line_number, word, pos = line.split('\t')
            words.append(word)
            pos_list.append(pos)
        words.append(None)
        pos_list.append('STOP')
        if ':' in pos_list:
            # skip this sentence because it has a POS tag of ':'
            return None
        else:
            return [words, pos_list]  # elements are paired, so lists are same length
 
def get_sentence_test(sentence_lines):
    for line in sentence_lines:
        words = [None]
       
        for line in sentence_lines:
            if not line:
                continue
            line_number, word = line.split('\t')
            words.append(word)
        words.append(None)
        return [words]

def read_sentences_from_file(fn, all_sentences_info):
    fin = open(fn,'r')
    sentence_lines = []
    sentence_count = 0
    line = fin.readline()
    line_count = 1
    while line: 
        line = line.strip()  # remove \n
        #print ('L:', line)
        if line:
            n,word,pos = line.split('\t')
            if pos ==':':
                print ('49:  pos:', pos, 'at line #', line_count)
            if word =='.' and pos =='.':
                if sentence_lines:
                    sent_info = get_sentence_info(sentence_lines)  # ------> 
                    if sent_info:
                        all_sentences_info.append(sent_info)
                        sentence_count += 1
                    sentence_lines = []  # reset for next sentence
                else:
                    print('56: Empty sentence at line', line_count)
            else:
                # this line is part of the sentence
                sentence_lines.append(line)
            
        line = fin.readline()
        line_count += 1
        
    print ('Found', sentence_count, 'sentences')
    fin.close()
    return all_sentences_info

def read_sentences_from_test_file(ft, all_sentences_info):
    fin = open(ft,'r')
    sentence_lines = []
    sentence_count = 0
    line = fin.readline()
    line_count = 1
    while line: 
        line = line.strip()  # remove \n
        #print ('L:', line)
        if line:
            n,word = line.split('\t')
            if word =='.':
                # end of sentence
                if sentence_lines:
                    sent_info = get_sentence_test(sentence_lines)  # ------> 
                    # sent_info is a list [word_list, tag_list] where elements are paired
                    # note None is returned if ':' was in the pos tag list
                    if sent_info:
                        all_sentences_info.append(sent_info)
                        sentence_count += 1
                    sentence_lines = []  # reset for next sentence
                else:
                    print('56: Empty sentence at line', line_count)
            else:
                # this line is part of the sentence
                sentence_lines.append(line)
            
        line = fin.readline()
        line_count += 1
        
    print ('Found', sentence_count, 'sentences')
    fin.close()
    return all_sentences_info


def gen_transition_matrix(all_sentences_info):
    """ Transition Matrix
        count all pos tuples by going thru all_sentences_info
        return trans and pos_tags
        trans: square matrix, one col and row per tag, 
            ea element is the transition prob from row tag to col tag
        pos_tags: list of string tag names incl 'START' at [0], 'STOP' at [-1]
    """
    pos_tuple_counts = defaultdict(int)
    pos_tags = set()
    s_count = 0
    print ('in gen_transition_matrix there are', len(all_sentences_info), 'sentences to process')
    try:
        for word_list, pos_list in all_sentences_info:
            s_count += 1
            pos_tuples = zip(pos_list, pos_list[1:])  # get sequential pairs of pos tags
            for pos_tuple in pos_tuples:
                tag1, tag2 = pos_tuple
                pos_tags.add(tag1)
                pos_tags.add(tag2)
                pos_tuple_counts[pos_tuple] += 1
    except Exception as err:
        print ('81: ERROR:', err)
        print ('    s_count =', s_count)
    
    if DEBUG:
        print ('There are', len(pos_tuple_counts), 'POS tuples')
        print ()
            
        print ('-- POS tags --')
        print (pos_tags)
        print ('count: ', len(pos_tags))
        print () 
    
    # make the list of POS tags start with 'START' and end with 'STOP'
    tag_list = sorted(list(pos_tags))
    tag_list.remove('START')  # so we can re-insert at [0]
    tag_list.remove('STOP')
    tag_list.insert(0, 'START')
    tag_list.append('STOP')
    print ('118:  tag_list:', tag_list)
    print ('  tag count:', len(tag_list))
    
    # convert the pos tuple counts into a Transition matrix
    dim = len(tag_list)
    t_shape = (dim, dim)
    trans = np.zeros(t_shape)
    
    # calc total counts for each POS tag
    total_counts = defaultdict(int)
    for (tag1, tag2), count in pos_tuple_counts.items():
        total_counts[tag1] += count
        if tag1 == 'NN':
            print (tag1, tag2, count)
    print ('NN total count:', total_counts['NN'])
    
    # store index for tag
    idx = {}
    for n,tag in enumerate(tag_list):
        idx[tag] = n
        
    # use 'product' to form iterator over all pairs of tags
    all_tag_pairs = product(tag_list, repeat=2)
    for tag1, tag2 in all_tag_pairs:
        tag_pair_count = pos_tuple_counts[(tag1,tag2)]
        tag1_total_counts = total_counts[tag1]
        if tag1_total_counts == 0:
            prob = 0.0
        else:
            prob = tag_pair_count / tag1_total_counts
        trans[idx[tag1], idx[tag2]] = prob
    
    return trans, tag_list

    
def gen_emission_prob(all_sentences_info, tags_set):
    total_word_counts_by_tag = defaultdict(int)
    
    def get_dd():
        return defaultdict(int)
    word_counts_dicts_by_tag = defaultdict( get_dd )
    # word_counts_dict:
    #   key: word
    #   val: word count (only for given tag)
    
    all_known_words = set()
    
    # for a given tag:  count total words and count by word
    for word_list, pos_list in all_sentences_info:
        words = word_list[1:-1]  # strip start and end None
        tags = pos_list[1:-1]    # strip 'START' and 'STOP'
        for word, tag in zip(words, tags):
            all_known_words.add(word)
            total_word_counts_by_tag[tag] += 1
            word_counts = word_counts_dicts_by_tag[tag]
            word_counts[word] += 1
    
    # convert (per tag) counts per word and total word counts to word | tag prob
    tags_list = list(tags_set)
    tags = tags_list[1:-1]  # remove 'START', 'STOP'
    
    # dict of emission (word | tag) probability
    em_word_tag_prob = {}
    #  key: (word, tag)
    #  val: probability of word given tag
    
    for tag in tags:
        total_word_count_for_this_tag = total_word_counts_by_tag[tag]
        word_counts = word_counts_dicts_by_tag[tag]
        #print ('207  "food" in word_counts for tag', tag, 'food' in word_counts)
        for word, count in word_counts.items():
            em_word_tag_prob[(word,tag)] = count * 1.0 / total_word_count_for_this_tag

    return em_word_tag_prob, all_known_words
    
def test_get_sentence_info():
    sent_lines = """1    Comparison    O
2    with    O
3    alkaline    B
4    phosphatases    I
5    and    O
6    5    B
7    -    I
8    nucleotidase    I
9    .    O
    """
    
    info = get_sentence_info(sent_lines)
    for x in info:
        print (x)
        print ('len:',len(x))
   

def create_obs_matrix(sentence, pos_tag_list, em_word_tag_prob, all_known_words):
    """ em_word_tag_prob is dict, key: (word,tag),  val: probability 
        all_known_words is a set
        return obs_matrix   an np array shape (tag_count, word_count)
    """
    #pos_tag_list.remove('START')
    #assert 'STOP' not in pos_tag_list
    
    if DEBUG:
        print ('===== in create_obs_matrix ========')
        print ('sentence:', sentence)
        print ('pos_tag list:', pos_tag_list)
        print ('len em_word_tag_prob:', len(em_word_tag_prob))
        print
    word_list = sentence.split()    
    obs_shape = ( len(pos_tag_list), len(word_list) )  # rows, cols
    obs_matrix = np.zeros(obs_shape)
    
    # put unknown words into em_word_tag_prob as 'NNP', 1.0
    for word in word_list:
        if word not in all_known_words:
            all_known_words.add(word)
            em_word_tag_prob[(word,'NNP')] = 1.0  # 100% prob it's NNP
    
    for row, tag in enumerate(pos_tag_list):
        for col, word in enumerate(word_list):
            if (word,tag) in em_word_tag_prob:
                prob = em_word_tag_prob[(word, tag)]
            else:
                prob = 0
            if DEBUG and prob > 0:
                print ('31  ', word, tag, prob)
            obs_matrix[row,col] = prob
    
    return obs_matrix


def viterbi_decode2(trans_array, em_array, pos_tag_list, sentence):
    """
    trans_matrix (tag transition matrix) is an np array
    em is the emission (observations) matrix an np array shape (tag_count, word_count)
    """
    if DEBUG:
        print ()
        print ('===== in viterbi_decode ========')
        print ('SENT:', sentence)
        print()
    
    words = sentence.split()
    # START row to find max prob 
    tag_index = 0
    real_tags = pos_tag_list[:]
    real_tags.remove('START')
    if 'STOP' in real_tags:
        real_tags.remove('STOP')
    real_tag_count = len(real_tags)
    
    # mpp  max path prob is np array (tag count x word counts)
    mpp   = np.zeros( (real_tag_count, len(words)) )
    bkptr = np.zeros( (real_tag_count, len(words)), dtype=int )
    #print('mpp shape', mpp.shape)
    
    # trim trans
    # remove bottom row of 'STOP' probs
    # remove col 0 (START) and col -1 (STOP)
    trans = trans_array[:-1, 1:-1]
    #print ('trans shape', trans.shape)
    
    # trim em
    # remove top row (START) and bottom row (STOP)
    em = em_array[1:-1,:]
    #print ('em shape', em.shape)
    
    if DEBUG:
        print ('trimmed trans')
        #print_array_as_lists(trans)
        print('  shape', trans.shape)
        print()
        
        print('trimmed em')
        print(em)
        print('  shape', em.shape)
        print()
        
        print('real tag list')
        print(real_tags)
        print('  len ', len(real_tags))
        print()    
    
    """ init the first mpp cols """ 
    #  prob for tagn = p START:tagn * em(20 | tagn)
    #  k is word position
    #  REM  [:,c] is col c,   [r,:] is row r
    k = 0
    idx_of_start = 0
    ##mpp[:,k] = trans[idx_of_start,:] * em[:,k]
    print('trans shape:', trans.shape, '  em shape', em.shape)
    xxx = np.log(em[:,k])
    mpp[:,k] = np.log(trans[idx_of_start,:]) + np.log(em[:,k])
    #print ('mpp')
    #print (mpp)
    
    """ Calc probs for k = 1,n for n words """
    for k in range(1, len(words)):
        ##for itag in range(1, len(real_tags) ):  # start a 1 to skip START tag row
        for itag in range(0, len(real_tags) ): 
            # find max( prob of prev paths to this tag )
            #probs_for_paths_to_this_tag = \
                #mpp[:,k-1] * trans[itag,:] * em[itag,k]
            '''
            probs_for_paths_to_this_tag = \
                mpp[:,k-1] * trans[1:,itag] * em[itag,k] # get trans col skipping row 0
            '''
            probs_for_paths_to_this_tag = \
                mpp[:,k-1] + np.log(trans[1:,itag]) + np.log(em[itag,k]) # get trans col skipping row 0

            max_prob = max(probs_for_paths_to_this_tag)
            mpp[itag,k] = max_prob
            
            ibkptr = np.argmax(probs_for_paths_to_this_tag)
            bkptr[itag,k] = ibkptr
            
            if DEBUG:
                print()
                print('k=', k,'  itag=', itag)
                print('trans for itag', itag)
                #print(trans[itag,:])
                print(trans[1:,itag])
                print('em[itag,k]', em[itag,k])
                print('probs for itag', itag, probs_for_paths_to_this_tag)    
                if max_prob > 0.0:
                    print('    max prob:', max_prob, '  <================')
                    print('    argmax:', ibkptr)
                    print('mpp col',k,mpp[:,k])
                    print()
        
        ## TEMP
        if DEBUG:
            print()
            print('mpp')
            #print_array_as_lists(mpp)
            print()
            print('bkptrs')
            print(bkptr)
            print('-----------------------------')
            print()
            
    
    if DEBUG:
        print ('mpp')
        for row in mpp:
            for x in row:
                print('%8.3f ' % x, end='')
            print()
        print()
        print('bkptr')
        print(bkptr)
    
    # find max prob tag indexes
    tag_indexes = []
    final_tag_index = np.argmax(mpp[:,-1])
    tag_indexes.append(final_tag_index)
    itag = final_tag_index
    for col in range(-1,len(words)*-1, -1):
        #print ('col', col)
        itag = bkptr[itag,col]
        tag_indexes.append(itag)
    
    tag_indexes.reverse()
    #print('tag indexes')
    #print(tag_indexes)
    #print()
    tags = [real_tags[i] for i in tag_indexes]
    #print ('tags', tags)
    return tags
    


# --- main ----
fn = 'gene-trainF17.txt'
ft = 'gene-test.txt'

print ('Process', fn)
all_sentences_info = []
all_sentences_info = read_sentences_from_file(fn, all_sentences_info)
print ()

all_sentences_test = []
all_sentences_test = read_sentences_from_test_file(ft, all_sentences_test)
print()
#print ('-- sentence info --')
#for s in all_sentences_info:
    #print (s)
#print()

# Transitiona Matrix    is an np array
trans_matrix, pos_tag_list = gen_transition_matrix(all_sentences_info)  # ----->
print()
print ('---- pos_tag_list ---')
print(pos_tag_list)
print()

print ('---- transition matrix ---')
print_tagged_array(trans_matrix, 1000, pos_tag_list)
print ('   trans matrix shape', trans_matrix.shape)
print ()

# Emission Prob
print ()
print ('289  Calling gen_emission_prob() ... ')
em_word_tag_prob, all_known_words = gen_emission_prob(all_sentences_info, pos_tag_list)  # --> 


passed_count = 0
not_passed_count = 0
duplicate_count = 0
sentences_set = set()

ferr = None
#ferr = open('error_sentences.txt','w')
CRLF = '\n'

fout = open('matonte-michael-assgn2-test-output.txt','w')

def output_sentence_and_tags(fout, words, tags):
    for n, (word, tag) in enumerate(zip(words,tags)):
        ln = n+1
        line = '%d\t%s\t%s\n' % (ln, word, tag)
        fout.write(line)
    line = '%d\t%s\t%s\n\n' % (ln+1, '.', '.') # output final line and blank line
    fout.write(line)

print('==============  start test sentences =====================')

for n, words_list in enumerate(all_sentences_test):
    print('all_sentences_test')
    print (all_sentences_test[0])
    print(all_sentences_test[1])
    print('WORDS', words_list)
    
    words = words_list[0]
    words = words[1:-1]  # strip first and last None
    sentence = ' '.join(words)
    
    if sentence in sentences_set:
        duplicate_count += 1
        #continue
    sentences_set.add(sentence)
    print ('-----------------------------------')
    print(sentence)
    print ('-----------------------------------')    
    print()


    # create Observation matrix,  
    print ('//////////////////////////////')
    print('SENT', sentence)
    print('TAGS', pos_tag_list)
    #print('em_word_tag_prob', em_word_tag_prob)
    
    obs_matrix = create_obs_matrix(sentence, pos_tag_list, em_word_tag_prob, all_known_words) #------>
    """ an np array shape (tag_count, word_count) """
    if True:
        print ('SENTENCE:',sentence)
        #print (sentence_tags)
        print ('--- obs matrix ---')
        print('obs matrix shape:', obs_matrix.shape)
        print_array(obs_matrix, 1000)
        print ('  shape:', obs_matrix.shape)
    
    est_tags = viterbi_decode2(trans_matrix, obs_matrix, pos_tag_list, sentence) #------>
    print()
    print('TAGS', est_tags)
    print()
    
    # output to file
    output_sentence_and_tags(fout, words, est_tags)
    
if fout:
    fout.close()
if ferr:
    ferr.close()