# Core Packages
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# EDA Packages

st.set_page_config(page_title='GAD Analysis', page_icon='images/logo.png',
                   layout='wide', initial_sidebar_state='auto')
sns.set(rc={'figure.figsize': (20, 15)})

DATA_URL = ('dataset/gad.csv')

st.markdown('# Graduate Admission Dataset')
st.markdown('### **Analysis of Graduate Admission Dataset**')

img = Image.open('images/gad.png')
st.image(img, width=720, caption='Graduate Admission Dataset')

st.markdown('### **About the Dataset:**')
st.info('This dataset was built \
    with the purpose of helping students in \
        shortlisting universities with their profiles. \
            The predicted output gives them a fair \
                idea about their chances for a particular university. \
                    This dataset is inspired by the UCLA Graduate Dataset from Kaggle. \
                        The graduate studies dataset is a dataset which describes the probability of \
                            selections for Indian students dependent on the following parameters below.')

img = Image.open('images/univ.png')
st.image(img, width=720, caption="Top 5 Universities in the US")

st.markdown('### **Dataset Info:**')
st.markdown('##### **Attributes of the Dataset:**')
st.info('\t 1. GRE Score (out of 340), \
        \n\t 2. TOEFL Score (out of 120), \
        \n\t 3. University Rating (out of 5), \
        \n\t 4. Statement of Purpose/ SOP (out of 5), \
        \n\t 5. Letter of Recommendation/ LOR (out of 5), \
        \n\t 6. Research Experience (either 0 or 1), \
        \n\t 7. CGPA (out of 10), \
        \n\t 8. Chance of Admittance (ranging from 0 to 1)')

img = Image.open('images/par.png')
st.image(img, width=720, caption="Influence of the Attributes on the Dataset")


def load_data(nrows):
    df = pd.read_csv(DATA_URL, nrows=nrows)
    def lowercase(x): return str(x).lower()
    df.set_index('Serial No.', inplace=True)
    df.rename(lowercase, axis='columns', inplace=True)
    return df


st.title('Lets explore the Graduate Admission Dataset')
# Creating a text element and let the reader know the data is loading.

data_load_state = st.text('Loading graduate admissions dataset...')
# Loading 500 rows of data into the dataframe.
df = load_data(500)

# Notifying the reader that the data was successfully loaded.
data_load_state.text('Loading graduate admissions dataset...Completed!')

# Explore Dataset
st.header('Quick  Explore')
st.sidebar.subheader('Quick  Explore')
st.markdown("Tick the box on the side panel to explore the dataset.")

if st.sidebar.checkbox("Show Raw Data"):
    st.subheader('Raw data')
    st.write(df)

if st.sidebar.checkbox('Dataset Quick Look'):
    st.subheader('Dataset Quick Look:')
    st.write(df.head())

if st.sidebar.checkbox("Show Columns"):
    st.subheader('Show Columns List')
    all_columns = df.columns.to_list()
    st.write(all_columns)

if st.sidebar.checkbox('Statistical Description'):
    st.subheader('Statistical Data Descripition')
    st.write(df.describe())

if st.sidebar.checkbox('Missing Values?'):
    st.subheader('Missing values')
    st.write(df.isnull().sum())

st.header('Data Visualization')
st.markdown("Tick the box on the side panel to create your own Visualization.")
st.sidebar.subheader('Data Visualization')

if st.sidebar.checkbox('Count Plot'):
    st.subheader('Count Plot')
    st.info("If error, please adjust column name on side panel.")
    column_count_plot = st.sidebar.selectbox(
        "Choose a column to plot count.", df.columns[:5])
    fig = sns.countplot(x=column_count_plot, data=df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

if st.sidebar.checkbox('Distribution Plot'):
    st.subheader('Distribution Plot')
    st.info("If error, please adjust column name on side panel.")
    column_dist_plot = st.sidebar.selectbox(
        'Choose a column to plot density.', df.columns[:5])
    fig = sns.distplot(df[column_dist_plot])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Showing the Prediction Model
st.header('Building Prediction Model')
st.sidebar.subheader('Prediction Model')
st.markdown("Tick the box on the side panel to run Prediction Model.")


if st.sidebar.checkbox('View Prediction Model'):
    st.subheader('Prediction Model')
    pickle_in = open('models/Multiple_Linear_Regression.pkl', 'rb')
    model = pickle.load(pickle_in)

    @st.cache()
    # defining the function to predict the output
    def convert_toefl_to_ielts(val):
        if val > 69 and val < 94:
            score = 6.5
        if val > 93 and val < 102:
            score = 7.0
        if val > 101 and val < 110:
            score = 7.5
        if val > 109 and val < 115:
            score = 8.0
        if val > 114 and val < 118:
            score = 8.5
        if val > 117 and val < 121:
            score = 9.0
        return score

    def pred(gre, toefl, sop, lor, cgpa, resc, univ_rank):

        # Preprocessing user input
        # ielts = convert_toefl_to_ielts(toefl)

        if resc == 'Yes':
            resc = 1
        else:
            resc = 0

        #Predicting the output
        prediction = model.predict(
            [[gre, toefl, univ_rank, sop, lor, cgpa, resc]])

        st.info("Chance of Admittance for University Rank " + str(univ_rank) + " = " + str(prediction[0]*100) +" %")
                # str(prediction[0]*100))
        if prediction[0] >= 0.6667:
            st.success(
                'Congratulations! You are eligible to apply for this university!')
            chance = Image.open('images/chance.png')
            st.image(chance, width=300, caption="High Chances !")
        else:
            st.caption('Better Luck Next Time :)')
            no_chance = Image.open('images/noChance.jpg')
            st.image(no_chance, width=300, caption="Low Chances :(")

    # Main function for the UI of the webpage
    def main():

        # Text boxes in which user can enter data required to make prediction
        gre = st.number_input('GRE Score (out of 340):', min_value=0, max_value=340, value=260, step=1)
        toefl = st.number_input('TOEFL Score (out of 120):', min_value=0, max_value=120, value = 80, step=1)
        sop = st.slider("SOP Score (out of 5):", value=0.0,
                        min_value=0.0, max_value=5.0, step=0.5)
        lor = st.slider("LOR Score (out to 5):", value=0.0,
                        min_value=0.0, max_value=5.0, step=0.5)
        resc = st.selectbox('Research Experience:', ("Yes", "No"))
        cgpa = st.number_input('Enter CGPA (out of 10):', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        univ_rank = st.slider("University Rank (1 to 5):", value=1,
                        min_value=1, max_value=5, step=1)

        # when 'Predict' is clicked, make the prediction and store it
        if st.button("Predict"):
            result = pred(gre, toefl, sop, lor, cgpa, resc, univ_rank)

    if __name__ == '__main__':
        main()

st.sidebar.subheader('Data Source')
st.sidebar.info("[Kaggle : graduate-admissions](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)")
st.sidebar.subheader('Author Credits')
st.sidebar.info("[Eswaramoorty K](https://github.com/eswar-2001)\
    \n [Krishnan S](https://github.com/skrishnan2001)\
    \n [Rohith S](https://github.com/Rohith8ceg)\
    \n [Vishnu Vasan R](https://github.com/vishnu-vasan)")
st.sidebar.subheader('Built with Streamlit')
st.sidebar.info("https://www.streamlit.io/")
