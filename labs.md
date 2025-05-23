# Generative AI for Developers Deep Dive
## Understanding key Gen AI concepts - full-day workshop
## Session labs 
## Revision 4.1 - 04/06/25

**Follow the startup instructions in the README.md file IF NOT ALREADY DONE!**

**NOTE: To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V. Chrome may work best for this.**

**Lab 1 - Working with Neural Networks**

**Purpose: In this lab, we’ll learn more about neural networks by seeing how one is coded and trained.**

1. In our repository, we have a set of Python programs to help us illustrate and work with concepts in the labs. These are mostly in the *genai* subdirectory. Go to the *TERMINAL* tab in the bottom part of your codespace and change into that directory.
```
cd genai
```

2. For this lab, we have a simple neural net coded in Python. The file name is nn.py. Open the file either by clicking on [**genai/nn.py**](./genai/nn.py) or by entering the command below in the codespace's terminal.

```
code nn.py
```

3. Scroll down to around line 55. Notice the *training_inputs* data and the *training_outputs* data. Each row of the *training_outputs* is what we want the model to predict for the corresponding input row. As coded, the output for the sample inputs ends up being the same as the first element of the array.  For inputs [0,0,1] we are trying to train the model to predict [0]. For the inputs [1,0,1], we are trying to train the model to predict [1], etc. The table below may help to explain.

| **Dataset** | **Values** | **Desired Prediction** |
| :---------: | :--------: | :--------------------: |
| **1** |  0  0  1  |            0           |
| **2** |  1  1  1  |            1           |
| **3** |  1  0  1  |            1           |
| **4** |  0  1  1  |            0           |

4. When we run the program, it will train the neural net to try and predict the outputs corresponding to the inputs. You will see the random training weights to start and then the adjusted weights to make the model predict the output. You will then be prompted to put in your own training data. We'll look at that in the next step. For now, go ahead and run the program (command below) but don't put in any inputs yet. Just notice how the weights have been adjusted after the training process.

```
python nn.py
```
![Starting run of simple nn](./images/gaidd30.png?raw=true "Starting run of simple nn") 

5. What you should see is that the weights after training are now set in a way that makes it more likely that the result will match the expected output value. (The higher positive value for the first weight means that the model has looked at the training data and realized it should "weigh" the first input higher in its prediction.) To prove this out, you can enter your own input set - just use 1's and 0's for each input. 

![Inputs to simple nn](./images/gaidd31.png?raw=true "Inputs to simple nn") 

6. After you put in your inputs, the neural net will process your input and because of the training, it should predict a result that is close to the first input value you entered (the one for *Input one*).

![Prediction close to first input](./images/gaidd32.png?raw=true "Prediction close to first input") 

7. Now, let's see what happens if we change the expected outputs to be different. In the editor for the genai_nn.py file, find the line for the *training_outputs*. Modify the values in the array to be ([[0],[1],[0],[1]]). These are the values of the second element in each of the training data entries. After you're done, save your changes as shown below, or use the keyboard shortcut.

![Modifying expected outputs](./images/gaidd33.png?raw=true "Modifying expected outputs")
![Saving changes](./images/gaidd9.png?raw=true "Saving changes")

8. Now, run the neural net again. This time when the weights after training are shown, you should see a bias for a higher weight for the second item.
```
python nn.py
```
![Second run of simple nn](./images/gaidd34.png?raw=true "Second run of simple nn") 

9. At the input prompts, just input any sequence of 0's and 1's as before.

10. When the trained model then processes your inputs, you should see that it predicts a value that is close to 0 or 1 depending on what your second input was.

![Second output of simple nn](./images/gaidd35.png?raw=true "Second output of simple nn")

11. (Optional) If you get done early and want more to do, feel free to try other combinations of training inputs and training outputs.
    
<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 2 - Experimenting with Tokenization**

**Purpose: In this lab, we'll see how different models do tokenization.**

1. In the same *genai* directory, we have a simple program that can load a model and print out tokens generated by it. The file name is *tokenizer.py*. You can view the file either by clicking on [**genai/tokenizer.py**](./genai/tokenizer.py) or by entering the command below in the codespace's terminal (assuming you're still in the *genai* directory).

```
code tokenizer.py
```
2. This program can be run and passed a model to use for tokenization. To start, we'll be using a model named *bert-base-uncased*. Let's look at this model on huggingface.co.  Go to https://huggingface.co/models and in the *Models* search area, type in *bert-base-uncased*. Select the entry for *google-bert/bert-base-uncased*.

![Finding bert model on huggingface](./images/gaidd12.png?raw=true "Finding bert model on huggingface")

3. Once you click on the selection, you'll be on the *model card* tab for the model. Take a look at the model card for the model and then click on the *Files and Versions* and *Community* tabs to look at those pages.

![huggingface tabs](./images/gaidd13.png?raw=true "huggingface tabs")

4. Now let's switch back to the codespace and, in the terminal, run the *tokenizer* program with the *bert-base-uncased* model. Enter the command below. This will download some of the files you saw on the *Files* tab for the model in HuggingFace.
```
python tokenizer.py bert-base-uncased
```
5. After the program starts, you will be at a prompt to *Enter text*. Enter in some text like the following to see how it will be tokenized.
```
This is sample text for tokenization and text for embeddings.
```
![input for tokenization](./images/gaidd36.png?raw=true "input for tokenization")

6. After you enter this, you'll see the various subword tokens that were extracted from the text you entered. And you'll also see the ids for the tokens stored in the model that matched the subwords.

![tokenization output](./images/gaidd37.png?raw=true "tokenization output")

7. Next, you can try out some other models. Repeat steps 4 - 6 for other tokenizers like the following. (You can use the same text string or different ones. Notice how the text is broken down depending on the model and also the meta-characters.)
```
python tokenizer.py roberta-base
python tokenizer.py gpt2
python tokenizer.py xlnet-large-cased
```
8. (Optional) If you finish early and want more to do, you can look up the models from step 7 on huggingface.co/models.
   
<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 3 - Understanding embeddings, vectors and similarity measures**

**Purpose: In this lab, we'll see how tokens get mapped to vectors and how vectors can be compared.**

1. In the repository, we have a Python program that uses a Tokenizer and Model to create embeddings for three terms that you input. It then computes and displays the cosine similarity between each combination. Open the file to look at it by clicking on [**genai/vectors.py**](./genai/vectors.py) or by using the command below in the terminal.
```
code vectors.py
```
2. Let's run the program. As we did for the tokenizer example, we'll pass in a model to use. We'll also pass in a second argument which is the number of dimensions from the vector for each term to show. Run the program with the command below. You can wait to enter terms until the next step.
```
python vectors.py bert-base-cased 5
```
![vectors program run](./images/gaidd38.png?raw=true "vectors program run")

3. The command we just ran loads up the bert-base-cased model and tells it to show the first 5 dimensions of each vector for the terms we enter. The program will be prompting you for three terms. Enter each one in turn. You can try two closely related words and one that is not closely related. For example
   - king
   - queen
   - duck

![vectors program inputs](./images/gaidd39.png?raw=true "vectors program inputs")

4. Once you enter the terms, you'll see the first 5 dimensions for each term. And then you'll see the cosine similarity displayed between each possible pair. This is how similar each pair of words is. The two that are most similar should have a higher cosine similarity "score".

![vectors program outputs](./images/gaidd40.png?raw=true "vectors program outputs")

5. Each vector in the bert-based models have 768 dimensions. Let's run the program again and tell it to display 768 dimensions for each of the three terms.  Also, you can try another set of terms that are more closely related, like *multiplication*, *division*, *addition*.
```
python vectors.py bert-base-cased 768
```
6. You should see that the cosine similarities for all pair combinations are not as far apart this time.
![vectors program second outputs](./images/gaidd19.png?raw=true "vectors program second outputs")

7. As part of the output from the program, you'll also see the *token id* for each term. (It is above the print of the dimensions. If you don't want to scroll through all the dimensions, you can just run it again with a small number of dimensions like we did in step 2.) If you're using the same model as you did in lab 2 for tokenization, the ids will be the same. 

![token id](./images/gaidd20.png?raw=true "token id")

8. You can actually see where these mappings are stored if you look at the model on Hugging Face. For instance, for the *bert-base-cased* model, you can go to https://huggingface.co and search for bert-base-cased. Select the entry for google-bert/bert-base-cased.

![finding model](./images/gaidd21.png?raw=true "finding model")

8. On the page for the model, click on the *Files and versions* tab. Then find the file *tokenizer.json* and click on it. The file will be too large to display, so click on the *check the raw version* link to see the actual content.

![selecting tokenizer.json](./images/gaidd22.png?raw=true "selecting tokenizer.json")
![opening file](./images/gaidd23.png?raw=true "opening file")

9. You can search for the terms you entered previously with a Ctrl-F or Cmd-F and find the mapping between the term and the id. If you look for "##" you'll see mappings for parts of tokens like you may have seen in lab 2.

![finding terms in file](./images/gaidd24.png?raw=true "finding terms in files")

10. If you want, you can try running the *genai_vectors.py* program with a different model to see results from other models (such as we used in lab 2) and words that are very close like *embeddings*, *tokenization*, *subwords*.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 4 - Working with transformer models**

**Purpose: In this lab, we’ll see how to interact with various models for different standard tasks**

1. In our repository, we have several different Python programs that utilize transformer models for standard types of LLM tasks. One of them is a simple translation example. The file name is *translation.py*. Open the file either by clicking on [**genai/translation.py**](./genai/translation.py) or by entering the command below in the codespace's terminal.

```
code translation.py
```
2. Take a look at the file contents.  Notice that we are pulling in a specific model ending with 'en-fr'. This is a clue that this model is trained for English to French translation. Let's find out more about it. In a browser, go to *https://huggingface.co/models* and search for the model name 'Helsinki-NLP/opus-mt-en-fr' (or you can just go to huggingface.co/Helsinki-NLP/opus-mt-en-fr).
![model search](./images/gaidd26.png?raw=true "model search")

3. You can look around on the model card for more info about the model. Notice that it has links to an *OPUS readme* and also links to download its original weights, translation test sets, etc.

4. When done looking around, go back to the repository and look at the rest of the *translation.py* file. What we are doing is loading the model, the tokenizer, and then taking a set of random texts and running them through the tokenizer and model to do the translation. Go ahead and execute the code in the terminal via the command below.
```
python translation.py
```
![translation by model](./images/gaidd41.png?raw=true "translation by model")
 
5. There's also an example program for doing classification. The file name is classification.py. Open the file either by clicking on [**genai/classification.py**](./genai/classification.py) or by entering the command below in the codespace's terminal.

```
code classification.py
```
6. Take a look at the model for this one *joeddav/xlm-roberta-large-xnli* on huggingface.co and read about it. When done, come back to the repo.

7. This uses a HuggingFace pipeline to do the main work. Notice it also includes a list of categories as *candidate_labels* that it will use to try and classify the data. Go ahead and run it to see it in action. (This will take awhile to download the model.) After it runs, you will see each topic, followed by the ratings for each category. The scores reflect how well the model thinks the topic fits a category. The highest score reflects which category the model thinks fit best.
```
python classification.py
```
![classification by model](./images/gaidd42.png?raw=true "classification by model")

8. Finally, we have a program to do sentiment analysis. The file name is sentiment.py. Open the file either by clicking on [**genai/sentiment.py**](./genai/sentiment.py) or by entering the command below in the codespace's terminal.

```
code sentiment.py
```

9. Again, you can look at the model used by this one *distilbert-base-uncased-finetuned-sst-2-english* in Hugging Face.

10. When ready, go ahead and run this one in the similar way and observe which ones it classified as positive and which as negative and the relative scores.
```
python sentiment.py
```
![sentiment by model](./images/gaidd43.png?raw=true "sentiment by model")

11. If you're done early, feel free to change the texts, the candidate_labels in the previous model, etc. and rerun the models to see the results.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 5 - Using Ollama to run models locally**

**Purpose: In this lab, we’ll start getting familiar with Ollama, a way to run models locally.**

1. We already have a script that can download and start Ollama and fetch some models we'll need in later labs. Take a look at the commands being done in the *../scripts/startOllama.sh* file. 
```
cat ../scripts/startOllama.sh
```

2. Go ahead and run the script to get Ollama and start it running.
```
../scripts/startOllama.sh &
```

The '&' at the end will causes the script to run in the background. You will see a set of startup messages. After those, you can just hit *Enter* to get back to a prompt in the terminal.

![starting ollama](./images/gaidd44.png?raw=true "starting ollama")

3. Now let's find a model to use.
Go to https://ollama.com and in the *Search models* box at the top, enter *llava*.
![searching for llava](./images/dga39.png?raw=true "searching for llava")

4. Click on the first entry to go to the specific page about this model. Scroll down and scan the various information available about this model.
![reading about llava](./images/dga40a.png?raw=true "reading about llava")

5. Switch back to a terminal in your codespace. While it's not necessary to do as a separate step, first pull the model down with ollama. (This will take a few minutes.)
```
ollama pull llava
```
6. Now you can run it with the command below.
```
ollama run llava
```
7. Now you can query the model by inputting text at the *>>>Send a message (/? for help)* prompt. Since this is a multimodal model, you can ask it about an image too. Try the following prompt that references a smiley face file in the repo.
```
What's in this image?  ../samples/smiley.jpg
```
(If you run into an error that the model can't find the image, try using the full path to the file as shown below.)
```
What's in this image? /workspaces/genai-dd/samples/smiley.jpg
```
![smiley face analysis](./images/gaidd45.png?raw=true "Smiley face analysis")

8. Now, let's try a call with the API. You can stop the current run with a Ctrl-D or switch to another terminal. Then put in the command below (or whatever simple prompt you want). 
```
curl http://localhost:11434/api/generate -d '{
  "model": "llava",
  "prompt": "What causes wind?",
  "stream": false
}'
```

9. This will take a minute or so to run. You should see a single response object returned. You can try out some other prompts/queries if you want.

![query response](./images/gaidd46.png?raw=true "Query response")

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 6 - Working with Vector Databases**

**Purpose: In this lab, we’ll learn about how to use vector databases for storing supporting data and doing similarity searches.**

1. In our repository, we have a simple program built around a popular vector database called Chroma. The file name is vectordb.py. Open the file either by clicking on [**genai/vectordb.py**](./genai/vectordb.py) or by entering the command below in the codespace's terminal.

```
code vectordb.py
```

2. For purposes of not having to load a lot of data and documents, we've *seeded* the same data strings in the file that we're loosely referring to as *documents*. These can be seen in the *datadocs* section of the file.
![data docs](./images/gaidd47.png?raw=true "Data docs")

3. Likewise, we've added the metadata again for categories for the data items. These can be seen in the *categories* section.
![data categories](./images/gaidd48.png?raw=true "Data categories")

4. Go ahead and run this program using the command shown below. This will take the document strings, create embeddings and vectors for them in the Chroma database section and then wait for us to enter a query.
```
python vectordb.py
```
![waiting for input](./images/gaidd49.png?raw=true "Waiting for input")

5. You can enter a query here about any topic and the vector database functionality will try to find the most similar matching data that it has. Since we've only given it a set of 10 strings to work from, the results may not be relevant or very good, but represent the best similarity match the system could find based on the query. Go ahead and enter a query. Some sample ones are shown below, but you can choose others if you want. Just remember it will only be able to choose from the data we gave it. The output will show the closest match from the doc strings and also the similarity and category.
```
Tell me about food.
Who is the most famous person?
How can I learn better?
```
![query results](./images/gaidd50.png?raw=true "Query results")

6. After you've entered and run your query, you can add another one or just type *exit* to stop.

7. Now, let's update the number of results that are returned so we can query on multiple topics. In the file *vectordb.py*, change line 70 to say *n_results=3,* instead of *n_results=1,*. Make sure to save your changes afterwards.

![changed number of results](./images/gaidd51.png?raw=true "Changed number of results")

8. Run the program again with *python vectordb.py*. Now you can try more complex queries or try multiple queries (separated by commas). 

![multiple queries](./images/gaidd52.png?raw=true "Multiple queries")
 
9. When done querying the data, if you have more time, you can try modifying or adding to the document strings in the file, then save your changes and run the program again with queries more in-line with the data you provided. You can type in "exit" for the query to end the program.

10. In preparation for the next lab, remove the *llava* model and download the *llama3.2* model.
```
ollama rm llava
ollama pull llama3.2
```

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 7 - Working with RAG implemented with vector databases**

**Purpose: In this lab, we’ll build on the use of vector databases to parse a PDF and allow us to include it in context for LLM queries.**

1. In our repository, we have a simple program built for doing basic RAG processing. The file name is rag.py. Open the file either by clicking on [**genai/rag.py**](./genai/rag.py) or by entering the command below in the codespace's terminal.

```
code rag.py
```

2. This program reads in a PDF, parses it into chunks, creates embeddings for the chunks and then stores them in a vector database. It then adds the vector database as additional context for the prompt to the LLM. There is an example pdf named *data.pdf* in the *samples* directory. It contains the same random document strings that were in some of the other programs. You can look at it in the GitHub repo if interested. Open up https://github.com/skillrepos/genai-dd/blob/main/samples/data.pdf if interested.

3. You can now run the program and pass in the ../samples/data.pdf file. This will read in the pdf and tokenize it and store it in the vector database. (Note: A different PDF file can be used, but it needs to be one that is primarily just text. The PDF parsing being used here isn't sophisticated enough to handle images, etc.)
```
python rag.py ../samples/data.pdf
```
![reading in the pdf](./images/gaidd54.png?raw=true "Reading in the PDF")

4. The program will be waiting for a query. Let's ask it for a query about something only in the document. As a suggestion, you can try the one below. (This will take a few minutes to run typically. Also, the response you get may vary from what is shown.)
```
What does the document say about art and literature topics?
```
5. The response should include only conclusions based off the information in the document.
![results from the doc](./images/gaidd74.png?raw=true "Results from the doc")
  
6. Now, let's ask it a query for some extended information. For example, try the query below. Then hit enter.
```
Give me 5 facts about the Mona Lisa
```
7. In the data.pdf file, there is one (and only one) fact about the Mona Lisa - an obscure one about no eyebrows. In the output, you will probably see only this fact or you might see this one and others based on this one or noting a lack of other information. 

![5 facts about the Mona Lisa](./images/gaidd75.png?raw=true "5 facts about the Mona Lisa")
   
8. The reason the LLM couldn't add any other facts was due to the PROMPT_TEMPLATE we have in the *rag.py* file. Take a look at it starting around line 29. Note how it limits the LLM to only using the context that comes from our doc (line 51).

![prompt template](./images/rag30.png?raw=true "prompt template")

![doc context](./images/rag31.png?raw=true "doc context")

9. To change this so the LLM can use our context and its own training, we need to change the PROMPT_TEMPLATE. Replace the existing PROMPT_TEMPLATE at lines 29-37 with the lines below. Afterwards, your changes should look like the screenshot below.  (If you see a red wavy line after pasting in the new template, you may need to remove some indenting. You can click on the *PROBLEMS* tab in the same row as *TERMINAL* to see the specific issues.)
```
    PROMPT_TEMPLATE = """
    Answer the question: {question} using whatever resources you have.
    Include any related information from {context} as part of your answer but add additional information from the model.
    Provide a detailed answer.
    Don’t justify your answers.
    """
```
![new prompt template](./images/gaidd79.png?raw=true "new prompt template")

10. **Save your changes**. Type "exit" to end the current run and then run the updated code. Enter the same query "Give me 5 facts about the Mona Lisa". This time, the program will run for several minutes and then the LLM should return 5 "real" facts about the Mona Lisa with our information included. Notice the highlighted part of the fourth item in the screenshot below.  (If the answer isn't returned by the time the break is over, you can just leave it running and check back later.)  You may also see info about other items in the doc in the output.

```
python rag.py ../samples/data.pdf
```
</br></br>
```
Give me 5 facts about the Mona Lisa
```
</br></br>
![new output](./images/rag33.png?raw=true "new output")

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 8 - Creating a simple agent**

**Purpose: In this lab, we’ll learn about the basics of agents and see how tools are called.**

1. Let's start out by looking at a limitation of LLMs - answering questions about current situations. Using the command below, run llama3.2 directly from ollama.  When it starts, you'll be at a prompt as shown in the screenshot below.
```
ollama run llama3.2
```

![Running model directly](./images/gaidd68.png?raw=true "Running model directly") 

2. Enter a prompt like the one shown below to ask the model for the current weather in a location. You can pick whatever city and state or country you want.

```
What is the current weather in <city>, <country or state>?
```

3. Notice that the model doesn't have real-time access to current weather information. So instead, it tries to be helpful about ways to find that information out. When done interacting with the model, use *Ctrl+D* to stop the interactive mode.

![No current info](./images/gaidd69.png?raw=true "No current info") 

4. Let's create a simple agent to help with this task. For this lab, we have the outline of an agent in a file called *agent.py* in the *genai* directory. You can take a look at the code either by clicking on [**genai/agent.py**](./genai/agent.py) or by entering the command below in the codespace's terminal.
   
```
code agent.py
```

![Starting agent](./images/gaidd70.png?raw=true "Starting agent") 

5. As you can see, this outlines the steps the agent will go through without all the code. When you are done looking at it, close the file by clicking on the "X" in the tab at the top of the file.

6. Now, let's fill in the code. To keep things simple and avoid formatting/typing frustration, we already have the code in another file that we can merge into this one. Run the command below in the terminal.
   
```
code -d ../extra/lab8-code.txt agent.py
```

7. Once you have run the command, you'll have a side-by-side in your editor of the completed code and the agent.py file.
  You can merge each section of code into the agent.py file by hovering over the middle bar and clicking on the arrows pointing right. Go through each section, look at the code, and then click to merge the changes in, one at a time.

![Side-by-side merge](./images/gaidd71.png?raw=true "Side-by-side merge") 

8. When you have finished merging all the sections in, the files should show no differences. Save the changes simply by clicking on the "X" in the tab name.

![Merge complete](./images/gaidd72.png?raw=true "Merge complete") 

9. Now you can run your agent with the following command:

```
python agent.py
```

10. At the prompt, you can enter a weather-related query. Start by asking it the same query you gave directly to the model earlier - about the current weather.

![Running agent](./images/gaidd73.png?raw=true "Running agent") 
   
11. You'll see some of the messages from the model loading. Then, eventually, you should see a section showing the call to the function, the return value from the function, and the final answer from the run.

![Agent output](./images/gaidd80.png?raw=true "Agent output") 

12. Notice that the location supplied in the user query was converted into an appropriate latitude and longitude for the tool call by the LLM. Then the output of the tool run was converted to a user-friendly weather report as the final answer.

(Optional) If you get done early and want to play around, you can try another current weather query or even asking it a more general weather question. If you don't seem to get a response after the function is called, it may be due to the API limiting. Ctrl-C to cancel the run, wait a moment, and try again.
<p align="center">
**[END OF LAB]**
</p>
</br></br>
<p align="center">
**THANKS!**
</p>
