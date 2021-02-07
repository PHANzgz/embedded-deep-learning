import streamlit as st 
import numpy as np 

# Load pages
import live_demo
import under_the_hood

PAGES = {
    "Live Demo": live_demo,
    "Under the hood": under_the_hood
}

def main():
    st.sidebar.title("Multilabel classification on the edge")
    selection = st.sidebar.radio("", list(PAGES.keys()))

    page = PAGES[selection]

    #with st.spinner("Loading {} ...".format(selection)):
    page.write()

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This project was created by Enrique Phan to demonstrate the development
        and deployment of a deep neural network for multilabel classifcation on
        a microcontroller. You can learn more about this app and others in 
        [the github repository](https://github.com/PHANzgz/embedded-deep-learning).
        """
        )

if __name__ == "__main__":
    main()