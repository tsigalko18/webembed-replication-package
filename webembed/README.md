# Doc2Vec_embedding_for_html
This repo uses Doc2vec and embeddings to distinguish between clone and distinct html pages (states) <br>

The dataset used to train the model and the trained model can be found at the following link: https://e1.pcloud.link/publink/show?code=kZCvXzZN8GHDg1TydR2471NTPGLWFmyyu4V. <br> Add them to the git repository to run the scripts and reproduce the results. The scripts must be run from the same folder in which they are stored. <br>

Files description:<br>
- script/01.warc2html.py: extracts all htmls from the warc files adn stores them in the folder commoncrawl.org
- script/02.html_reorder.py: moves all htmls in commoncrawls.org and DS_Crawls in a single folder, called all_html
- script/03.generate_corpus.py: extracts the data from each html in all_html. Specifically, it extracts the html content, tags or both for each html, and stores the extracted data in state_name.content, state_name.tags, or state_name.content_tags for each feature.
- script/03a.generate_corpusGS.py: does the same as the above file, only for the GroundTruthModel folder/htmls
- script/04.generate_setsDS.py generates the train and test set for the model. Specifically, it removes all pairs of manually annotated states for the htmls in DS_Crawls from the training set, and puts them in the test set.
- script/04a.generate_setsGTM.py: does the same as the above file, only for the htmls in the GroundTruthModel folder. The difference with the script above is that the train set is genrated but never used.
- script/05.generate_line_sentence.py: creates LineSentences from the corpus used to train the model (for efficency purposes. For more information, see the method tarin() found at the following link: https://radimrehurek.com/gensim/models/doc2vec.html)
- script/06.train_model.py: trains the models. Specifically, a trained model is saved after 5 epochs, until reaching epochs = 300.
- script/07.classifier_scoresDS.py: generates classifier scores for different classifiers for all apps in DS_Crawls. 
- script/07a.classifier_scoresGTM.py: generates classifer scores for each app in GroundTruthModel. It computes the scores per app in order to compare the obtained results with the baselines.
- script/08.compute_statistics.py: computes the wilcox score and cohens'd for each f_score (f_score based on the 'Ensamble' only) 
- abstract_function_python/main.py: exposes python app
- abstract_function_python/abstract_function.py: implements the abstract function <br>

To expose the python app and be able to make java and python comunicate, the app must first be exposed. To do so, from ther terminal, go to the abstract_function_python folder, run 'export FLASK_APP=main' to expose main.py, then run 'flask run'. 
<br>
Notes in case the experiment is re-done from zero: <br>
- the warc files need to be downloaded sequentially and stored inside the folder commoncrawls.org for the code to work. They can be found at the following link: https://commoncrawl.org/the-data/get-started/ <br>
- the htmls in the original GroundTruthModel folder were extracted manually and stored in a folder with the same name. The whole folder can be found at the following link: https://zenodo.org/record/3385377 <br>
- some tables were created manually, others with the scripts. <br>
- some parts of code were commented out. De-comment to create the desired resource.
