def read_files(files_path):
  read_files = glob.glob(os.path.join (files_path,'comments*.csv'))
  np_array_values = []
  for files in read_files:
    comments_data = pd.read_csv(files, header = 0, engine='python')
    np_array_values.append(comments_data)
    print(files)

  merge_values = np.vstack(np_array_values)
  comments_data = pd.DataFrame(merge_values)
  comments_data.columns = ['post_id', 'post_by', 'post_text', 'post_published', 'comment_id', 'comment_by', 'is_reply','comment_message',
                          'comment_published', 'comment_like_count']
  print('Created a dataframe')


  return comments_data



def predict_lang(df,column):
  #! pip install fastlangid
  from fastlangid.langid import LID
  langid = LID()
  df['Language'] =langid.predict(df[column].astype(str))
  return df['Language']


def preprocess_data(data,name,lang):
  

    stop_words_ru = stopwords.words('russian')
    stop_words_en = stopwords.words('english')
    # Lowering the case of the words in the sentences
    #data[name]=data[name].to_string(na_rep='').lower()
    # Code to remove the Hashtags from the text
    data[name]=data[name].apply(lambda x:re.sub(r'\B#\S+','',str(x)))
    # Code to remove the links from the text
    data[name]=data[name].apply(lambda x:re.sub(r"http\S+", "", str(x)))
    # Code to remove the Special characters from the text 
    data[name]=data[name].apply(lambda x:' '.join(re.findall(r'\w+', str(x))))
    # Code to substitute the multiple spaces with single spaces
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    # Code to remove all the single characters in the text
    #data[name]=data[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
    data[name]=data[name].apply(lambda x:re.sub(r'\s+[а-яА-Я]\s+', ' ', str(x)))
    data[name]=data[name].apply(lambda x:re.sub(r'\s+[A-Za-z]\s+', ' ', str(x)))
    
    # Remove the twitter handlers
    data[name]=data[name].apply(lambda x:re.sub('@[^\s]+','',str(x)))
    data[name]=data[name].str.lower()
    if lang=='en':
      data[name] = data[name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words_en)]))
    #data[name]= [w for w in data[name] if not w.lower() in stop_words_en]
    if lang=='ru':
      data[name] = data[name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words_ru)])) 
    





def predict_toxicity(df,column):
  #!pip install "git+https://github.com/tqdm/tqdm.git@devel#egg=tqdm"
  from tqdm import tqdm
  from time import sleep
  from detoxify import Detoxify

  speech = []
  #model = Detoxify('multilingual', device = 'cuda')
  model = Detoxify('multilingual')

  for example in tqdm(df[column].values):
      speech.append(model.predict(example)) 

  toxicity_df=pd.DataFrame(speech)
  #toxicity_df.drop(columns={'level_0'}, inplace=True)
  #toxicity_df.reset_index(inplace= True)
  #df.drop(columns={'level_0'}, inplace=True)
  #df.reset_index(inplace= True)  
  toxicity_predicted_df=pd.concat([df, toxicity_df], axis=1) 
  return toxicity_predicted_df
