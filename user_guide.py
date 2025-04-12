import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image


# Language selector
with open("user_guide.svg", "r") as f:
    svg_content = f.read()

# OR better SVG rendering:
st.markdown(f"<div>{svg_content}</div>", unsafe_allow_html=True)
language = st.session_state.get("language", "English")

if language == "English":
    st.write("""
    # 🧑‍💻 User Guide

    Welcome to the **Customer Segmentation System**! This guide will walk you through using the system, understanding the models, and exploring the dataset.

    ---

    ## ⚙️ Model Preparation

    The system supports two clustering techniques:

    ### 🔹 Manual - RFM
    Clustering Customer based on expert methods/knowledge (e.g., recency, frequency, monetary, category).

    - `manual_rfm`: Manual customer segmentation based on RFM calculations and experience.

    ### 🔹 Kmeans - RFM
    Clustering Customer based on K-means which is a powerful and widely used algorithm for discovering hidden structures in data by grouping similar data points together.

    - `model_Kmeans_seg.pkl`: trained with K-means which is a powerful and widely used algorithm for discovering hidden structures in data by grouping similar data points together.

    ---

    ## 📦 Datasets

    ### 📁 Raw Data
    - `Products_with_Categories.csv`: Product metadata (name, category).
    - `Transactions.csv`: User-product interaction history.

    ### 📁 Merge Data
    - `data_segmentation_total.csv`: Transaction metadata for each customer.
  
    It would be convenient to merge the two datasets for processing.

    ---

    ## 🧭 Navigation

    The project is structured across several main pages:

    ### 🏠 Introduction
    Overview of the project, goals, and recommendation approaches.

    ### 📘 User Guide
    Detailed instructions for using the app and understanding the models.

    ### 📊 EDA – Exploratory Data Analysis
    Visual exploration of the dataset:
    - Popular product categories
    - Top purchased products
    - User activity patterns

    ### 🤖 Recommendation
    Try out the recommender system:
    - Select product or user
    - Choose model (Content-based or Collaborative)
    - Get top-N product recommendations

    ---

    ## 🔗 GitHub Repository

    Access the full code, models, and documentation here:  
    👉 [GitHub Repository](https://github.com/anatwork14/data_science_recommend_system.git)

    ---

    Enjoy exploring and recommending with Shopee Recommendation System! 🚀
    """)

elif language == "Tiếng Việt":
    st.write("""
    # 🧑‍💻 Hướng Dẫn Sử Dụng

    Chào mừng bạn đến với **Hệ thống Phân khúc khách hàng**! Hướng dẫn này sẽ giúp bạn sử dụng hệ thống, hiểu các mô hình gợi ý và khám phá tập dữ liệu.

    ---

    ## ⚙️ Chuẩn Bị Mô Hình

    Hệ thống hỗ trợ hai kỹ thuật gợi ý:

    ### 🔹 Manual - RFM
    Phân khúc khách hàng bằng thủ công dựa vào các tính toán RFM và kinh nghiệm (ví dụ: Khoản cách từ lần mua hàng gần nhất, Tần suất mua hàng, doanh số mua hàng...)

    - `manual_rfm`: Phân khúc khách hàng bằng thủ công dựa vào các tính toán RFM và kinh nghiệm

    ### 🔹 Kmeans - RFM
    K-means là một thuật toán mạnh mẽ và được sử dụng rộng rãi để khám phá cấu trúc ẩn trong dữ liệu thông qua việc phân nhóm các điểm dữ liệu tương tự lại với nhau

    - `model_Kmeans_seg.pkl`: K-means là chia một tập dữ liệu thành K cụm riêng biệt, trong đó mỗi điểm dữ liệu được gán cho cụm có trung tâm (centroid) gần nhất

    ---

    ## 📦 Tập Dữ Liệu

    ### 📁 Dữ liệu gốc
    - `Products_with_Categories.csv`: Thông tin sản phẩm (tên, danh mục).
    - `Transactions.csv`: Lịch sử tương tác giữa người dùng và sản phẩm.

    ### 📁 Dữ liệu đã kết nối
    - `data_segmentation_total.csv`: Transaction metadata for each customer.
  
    It would be convenient to merge the two datasets for processing.

    ---

    ## 🧭 Điều Hướng

    Dự án được chia thành các trang chính:

    ### 🏠 Giới thiệu
    Tổng quan dự án, mục tiêu và các phương pháp gợi ý.

    ### 📘 Hướng dẫn sử dụng
    Hướng dẫn chi tiết cách sử dụng ứng dụng và hiểu các mô hình.

    ### 📊 Phân tích dữ liệu (EDA)
    Khám phá dữ liệu qua biểu đồ:
    - Các danh mục phổ biến
    - Sản phẩm được mua nhiều nhất
    - Hành vi người dùng

    ### 🤖 Gợi ý sản phẩm
    Trải nghiệm hệ thống gợi ý:
    - Chọn sản phẩm hoặc người dùng
    - Chọn mô hình (nội dung hoặc cộng tác)
    - Nhận top-N sản phẩm gợi ý

    ---

    ## 🔗 Kho Mã Nguồn GitHub

    Truy cập mã nguồn, mô hình và tài liệu tại:  
    👉 [GitHub Repository](https://github.com/anatwork14/data_science_recommend_system.git)

    ---

    Chúc bạn trải nghiệm vui vẻ cùng Shopee Recommendation System! 🚀
    """)
