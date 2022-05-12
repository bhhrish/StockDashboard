import streamlit as st
import home_page
import visualiser_page
import predictor_page

PAGE_PY = {
    "Home": home_page,
    "Visualiser": visualiser_page,
    "Predictor": predictor_page
}

def main():
    st.set_page_config(layout='wide', page_icon=':chart_with_upwards_trend:')
    st.sidebar.title("Menu")
    choices = st.sidebar.radio("Navigate", list(PAGE_PY.keys()))
    PAGE_PY[choices].main()


if __name__ == "__main__":
    main()