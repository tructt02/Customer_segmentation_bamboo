import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image

# Load the image
image = Image.open('Introduction.png')

# Language selector (or get from session state if set globally)
language = st.session_state.get("language", "English")

# Display title and image
st.write("# Data Science Project")
st.image(image, caption='Customer Segmentation', use_container_width=True)

# Content for English
if language == "English":
    st.write("""
    ---
    #### 📦 Project Title: 
    - ***Convenience store*** Customer Segmentation

    #### 🧠 Project Description:
    - The project aims to classify customers based on their purchasing behavior using the RFM (Recency, Frequency, Monetary) method combined with clustering algorithms such as K-Means, GMM, and Hierarchical Clustering.

    - By segmenting customers into groups, the project enables businesses to gain a deeper understanding of each group’s characteristics, optimize marketing strategies, personalize customer experiences, and enhance overall business performance.

    #### ⚙️ Key Features:
    - Personalized Experience: Enhances customer satisfaction through individualized offerings.

    #### 📊 Tech Stack & Tools:
    - `Languages`: Python

    - `Libraries`: Pandas, NumPy, Scikit-learn, Kmeans, RFM

    - `Visualization`: Matplotlib, Seaborn

    - `Deployment`: Streamlit

    #### 📈 Outcomes:
    - Customer Insights: Provides in-depth understanding of distinct customer groups and their characteristics.

    - Business Performance Improvement: Drives efficiency and profitability by targeting the right customer groups.

    - Ready-to-integrate API or demo showcasing recommendation functionality.

    ***
    """)

# Content for Vietnamese
else:
    st.write("""
    ---
    #### 📦 Tên Dự Án: 
    - Hệ thống phân khúc khách hàng ***Cửa hàng tiện lợi***

    #### 🧠 Mô tả Dự Án:
    - Dự án nhằm phân loại khách hàng dựa trên hành vi mua sắm bằng phương pháp RFM (Recency, Frequency, Monetary) kết hợp với các thuật toán phân cụm như K-Means, GMM và Hierarchical Clustering. 

    - Shông qua việc phân nhóm khách hàng, dự án giúp doanh nghiệp hiểu rõ đặc điểm từng nhóm, tối ưu hóa chiến lược tiếp thị, cá nhân hóa trải nghiệm khách hàng và nâng cao hiệu quả kinh doanh.

    #### ⚙️ Tính Năng Chính:
    - Phân khúc khách hàng cho từng người dùng

    #### 📊 Công Nghệ & Công Cụ:
    - `Ngôn ngữ`: Python

    - `Thư viện`: Pandas, NumPy, Scikit-learn, SurPRISE, Underthesea, Cosine Similarity

    - `Trực quan hóa`: Matplotlib, Seaborn

    - `Triển khai`: Streamlit

    #### 📈 Kết Quả:
    - Cung cấp sự hiểu biết sâu sắc về các nhóm khách hàng riêng biệt và đặc điểm của họ.

    - Cải thiện hiệu suất kinh doanh: Thúc đẩy hiệu quả và lợi nhuận bằng cách nhắm mục tiêu đúng các nhóm khách hàng.

    - Tích hợp dễ dàng qua API hoặc bản demo.

    ***
    """)
