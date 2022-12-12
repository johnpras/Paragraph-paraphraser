from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import streamlit as st
import joblib
import pickle
import random

st.set_page_config(
    page_title="AI Paraphraser",
)

st.title('Paraphraser Bot')

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   The *AI Paraphraser Bot* app is an easy-to-use interface built in Streamlit for paraphrasing sentences and parahgraphs.
-   Based on a T5 Model for generating paraphrases of english sentences. Trained on the Google PAWS dataset.​
	    """
    )

    st.markdown("")

st.markdown("")


with st.form(key="my_form"):

    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        InputType = st.radio(
            "Choose your input",
            ["Sentence", "Paragraph"],
        )

        k = st.slider(
            "# of results",
            min_value=1,
            max_value=10,
            value=1,
        )
        
        topk = st.slider(
            "# Top-k sampling",
            min_value=1,
            max_value=256,
            value=200,
        )
        
        topp = st.slider(
            "# Top-p sampling",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
        )
        
        

    with c2:
            doc = st.text_area(
                "Paste your text below (max 256 words)",
                height=300,
            )
    
            MAX_WORDS = 256
            import re
            res = len(re.findall(r"\w+", doc))
            if res > MAX_WORDS:
                st.warning(
                    "⚠️ Your text contains "
                    + str(res)
                    + " words."
                    + " Only the first 256 words will be reviewed. Upgrade to the pro plan to increase the total words allowance! 😊"
                )
    
                doc = doc[:MAX_WORDS]
    
            submit_button = st.form_submit_button(label="Rephrase")

    
    cs, c1, c2, c3  = st.columns([2, 1.5, 1.5, 1.5])
    


if submit_button:
    st.subheader("Results:")
    
    with st.spinner(text="This may take a moment..."):
    
        #tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
        #model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to("cuda")
        
        #joblib.dump(model, 'paraphraser_model.pkl', compress=9)
        #tokenizer_file = 'paraphraser_tokenizer.pickle'
        #pickle.dump(tokenizer, open(tokenizer_file, 'wb'))
        
        tokenizer = pickle.load(open("paraphraser_tokenizer.pickle", 'rb'))
        model = joblib.load('paraphraser_model.pkl')
        
      
        if InputType == "Sentence":
            
            sentence = doc
            if len(sent_tokenize(sentence)) == 1:
            
                text =  "paraphrase: " + sentence + " </s>"
                
                encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
                input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
                
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_masks,
                    max_length=256,
                    do_sample=True,
                    top_k=topk,
                    top_p=topp,
                    early_stopping=True,
                    num_return_sequences=k
                )
                
                
                for index, output in enumerate(outputs):
                    line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    print(line)
                    st.write(index+1,line)
            else:
                st.write("Its not a sentence.")
         
        else:
            
            input_text = doc
            lista =[]

            def my_paraphrase(sentence):
              for sent in sent_tokenize(input_text):
                sentence =  "paraphrase: " + sent + " </s>"

                encoding = tokenizer.encode_plus(sentence,pad_to_max_length=True, return_tensors="pt")
                input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_masks,
                    max_length=256,
                    do_sample=True,
                    top_k=topk,
                    top_p=topp,
                    early_stopping=True,
                    num_return_sequences=k
                )

                for output in outputs:
                  line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                  lista.append(line)

                    
            my_paraphrase(input_text)
            
            final_res=[]
            for l in range(k):
                listab=[]
                final_list=[]
                j=0
                n=k
                for i in range(len(sent_tokenize(input_text))):
                  possible_choices = [item for item in lista[j:n] if item != ""]
                  rundfromlista = random.choice(possible_choices)
                  listab.append(rundfromlista)
                  for l in lista[j:n]:
                      if l == rundfromlista:
                          l == ""
                      break
                  n+=k
                  j+=k
    
                final_list.append(' '.join(listab))
                strtext = ' '.join(final_list)
                listab=[]
                final_res.append(strtext)  
  
            for index, i in enumerate(final_res):
                st.write(index+1, i)