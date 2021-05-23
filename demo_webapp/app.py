import streamlit as st 
import numpy as np 

# Load pages
import person_detection.live_demo
import person_detection.under_the_hood
import handwriting_rec.demo
import handwriting_rec.under_the_hood
import predictive_maint.demo

APP_NAMES = [ "Multilabel classification on the edge",
         "Handwriting recognition using an accelerometer",
         "Predictive maintenance(TODO)"]

PAGES_PD = {
    "Live Demo": person_detection.live_demo,
    "Under the hood": person_detection.under_the_hood
}

PAGES_HR = {
    "Demo": handwriting_rec.demo,
    "Under the hood": handwriting_rec.under_the_hood
}

PAGES_PM = {
    "Demo": predictive_maint.demo
}

PAGES = [PAGES_PD, PAGES_HR, PAGES_PM]

def main():

    # Application select
    st.sidebar.title("Embedded deep learning")
    app_selection = st.sidebar.radio("Select an application to explore", APP_NAMES)
    
    # Section select
    st.sidebar.subheader(app_selection)
    app_selection_ix = APP_NAMES.index(app_selection)
    page_selection = st.sidebar.radio("", list(PAGES[app_selection_ix].keys()), index=1)
    page = PAGES[app_selection_ix][page_selection]

    # Write selected page
    #with st.spinner("Loading {} ...".format(selection)):
    page.write()

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This project was created by Enrique Phan to demonstrate the development
        and deployment of different deep learning applications on microcontrollers. 
        You can check the source code and learn more in
        [the github repository](https://github.com/PHANzgz/embedded-deep-learning).
        """
        )

if __name__ == "__main__":
    main()