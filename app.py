import streamlit as st

# ✅ Basic app health check
st.set_page_config(page_title="Phishing Email Detector", layout="wide")
st.title("📧 Phishing Email Detector")

# ✅ Sidebar Navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a page", ["Home", "About"])

# ✅ Page logic
if page == "Home":
    st.subheader("Test Page")
    st.write("🚀 Streamlit is working!")
    if st.button("Click Me"):
        st.success("🎉 You clicked the button!")
elif page == "About":
    st.subheader("About this App")
    st.markdown("""
    This is a minimal test version of the **Phishing Email Detector** app.
    """)
    
st.markdown("---")
st.markdown("💡 Built with Streamlit | Developed by Van Tran")
