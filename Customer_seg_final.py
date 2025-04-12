import streamlit as st
import pandas as pd
import pickle
import numpy as np

#import scipy.sparse

###### Giao diện Streamlit ######
st.image('w05227-small.jpg', use_container_width =True)
# Get user's language choice
language = st.session_state.get("language", "English")
# function cần thiết
def lookup_member(df, member_number):
    result = df[df['Member_number'].astype(str) == str(member_number)]
    if result.empty:
        return f"Không tìm thấy thành viên với Member_number = {member_number}"
    return result

def assign_cluster_names(df):
    cluster_name_map = {
        0: "Khách hàng trung thành",
        1: "Khách hàng đã mất",
        2: "Khách hàng VIP",
        3: "Khách hàng có nguy cơ rời bỏ",
        4: "Khách hàng không hoạt động"
    }
    df['cluster_names'] = df['Cluster'].map(cluster_name_map)
    return df

def segment_customers_kmeans(data_rfm):
    # Load K-Means model
    try:
        with open('model_Kmeans_seg.pkl', 'rb') as f:
            model_Kmeans_seg = pickle.load(f)
    except FileNotFoundError:
        print("Error: 'model_Kmeans_seg.pkl' not found. Please ensure the model file exists.")
        return pd.DataFrame(columns=['Member_numb', 'Cluster', 'cluster_names'])  # Return an empty DataFrame
    # Chuẩn bị dữ liệu để dự đoán
    df_now = data_rfm[['Recency','Frequency','Monetary']].copy()
    # Dự đoán phân cụm
    model_predict = model_Kmeans_seg.predict(df_now)
    # Thêm thông tin cụm vào DataFrame
    df_now["Cluster"] = model_predict
    # Reset index để lấy 'Member_numb' làm cột
    df_now = df_now.reset_index()
    # Gán tên cho các cụm
    df_now = assign_cluster_names(df_now)
    return df_now


def Manual_segments(df, recency_col='Recency', frequency_col='Frequency', monetary_col='Monetary'):
    # Calculate RFM quartiles and assign labels
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)

    r_groups = pd.qcut(df[recency_col].rank(method='first'), q=4, labels=r_labels)
    f_groups = pd.qcut(df[frequency_col].rank(method='first'), q=4, labels=f_labels)
    m_groups = pd.qcut(df[monetary_col].rank(method='first'), q=4, labels=m_labels)

    # Create new columns R, F, M
    df = df.assign(R=r_groups.values, F=f_groups.values, M=m_groups.values)
    # Concat RFM quartile values to create RFM Segments
    def join_rfm(x):
        return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
    df['RFM_Segment'] = df.apply(join_rfm, axis=1)
    # Calculate RFM_Score
    df['rfm_score'] = df[['R', 'F', 'M']].sum(axis=1)
    # Reset index để lấy 'Member_numb' làm cột
    df= df.reset_index()
    # Manual Segmentation based on RFM_Score
    def segment_by_score(rfm_score):
        if rfm_score >= 10:
            return 'Khách hàng cao cấp'
        elif rfm_score >= 7:
            return 'Khách hàng tiềm năng'
        elif rfm_score >= 5:
            return 'Khách hàng trung bình'
        else:
            return 'Khách hàng cần kích hoạt'
    # Apply segmentation
    df['cluster_names'] = df['rfm_score'].apply(segment_by_score)

    return df


# === 🎯 Filtering Method Selection ===
st.markdown("## 🎯 Clustering Method" if language == "English" else "## 🎯 Phương pháp gợi ý")

filtering_method = st.selectbox(
    "🔍 Choose your clustering approach:" if language == "English"
    else "🔍 Chọn phương pháp phân cụm bạn muốn sử dụng:",
    ("🧠 Manual RFM", "🤝 RFM - Kmeans") if language == "English"
    else ("🧠 Phân khúc theo kinh nghiệm ", "🤝 Phân khúc dùng Kmeans-RFM")
)

# Display a description below the selection
if language == "English":
    if "Kmeans" in filtering_method:
        st.info("Customer clustering using K-means with 5 clusters")
    else:
        st.info("Customer clustering into 4 groups based on expert methods/knowledge.")
else:
    if "Kmeans" in filtering_method:
        st.info("Phân cụm khách hàng theo Kmeans với số nhóm bằng 5")
    else:
        st.info("Phân cụm khách hàng theo phương pháp chuyên gia với số nhóm bằng 4")
st.markdown("---")

custom_info = f"""
    <div style="background-color: #e8f4fd; padding: 10px 12px; border-radius: 16px; margin-bottom: 20px">
        <span style="color: #0a66c2; font-size: 16px;">
             <strong>📌 {"Tips" if language == "English" else "Gợi ý"}:</strong> 
            {"Customer clustering will be used to create segments. A selected user is segmented based on their consumer behavior, enabling the suggestion of tailored offers and promotions to stimulate spending and enhance customer engagement." 
            if language == "English" else "Phân cụm khách hàng sẽ được sử dụng để tạo ra các phân khúc. Đối với một người dùng được chọn, dựa trên hành vi tiêu dùng để phân khúc họ, từ đây có thể đề xuất các chương trình ưu đãi, khuyến mãi giúp kích thích tiêu dùng, gắn kết khách hàng."}
        </span>
    </div>
    """
st.markdown(custom_info, unsafe_allow_html=True)

# Đọc dữ liệu giao dịch
df_trans = pd.read_csv('data_segmentation_total.csv')
#df_trans[df_trans.duplicated(keep=False)]

# Lấy 15 sản phẩm
random_trans = df_trans.head(n=15)
st.session_state.random_trans = random_trans

# Hiển thị danh sách 15 khách hàng
st.write("List Fifty-five customers:" if language == "English" else "Danh sách 15 khách hàng:")
st.dataframe(st.session_state.random_trans)



# Feature Engineering
df_trans['Gross'] = df_trans.price * df_trans['items']
df_trans['Order_id'] = df_trans.index
df_trans['Date'] = pd.to_datetime(df_trans['Date'])

# RFM
Recency = lambda x : (df_trans['Date'].max().date() - x.max().date()).days
Frequency  = lambda x: len(x.unique())
Monetary = lambda x : round(sum(x), 2)

data_RFM = df_trans.groupby('Member_number').agg({'Date': Recency,
                                                 'Order_id': Frequency,
                                                 'Gross': Monetary })
data_RFM.columns = ['Recency', 'Frequency', 'Monetary']
data_RFM = data_RFM.sort_values('Monetary', ascending=False)

most_frequent_category = df_trans.groupby(['Member_number', 'Category']).size().reset_index(name='count')
most_frequent_category = most_frequent_category.loc[most_frequent_category.groupby('Member_number')['count'].idxmax(), ['Member_number', 'Category']]

# Load K-Means model
# with open('model_Kmeans_seg.pkl', 'rb') as f:
#     model_Kmeans_seg = pickle.load(f)

# df_now = data_RFM[['Recency','Frequency','Monetary']].copy()
# model_predict = model_Kmeans_seg.predict(df_now)
# df_now["Cluster"] = model_predict
# df_now = assign_cluster_names(df_now)
# df_now = df_now.reset_index()

# GỌI Model KMEANS hoặc Manual LÊN DÙNG
df_now = Manual_segments(data_RFM, recency_col='Recency', frequency_col='Frequency', monetary_col='Monetary') if (filtering_method == "🧠 Manual RFM" or filtering_method == "🧠 Phân khúc theo kinh nghiệm ") else segment_customers_kmeans(data_RFM)

# Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
if 'selected_ma_san_pham' not in st.session_state:
    st.session_state.selected_ma_san_pham = None
# Gán st.session_state.random_trans vào một biến tạm để ngăn hiển thị
_ = st.session_state.random_trans  # Dùng "_" để gán, không hiển thị

#
st.markdown(
f"""
<div style='font-size: 1.25rem; font-weight: 600; margin-bottom: 0rem;'>
    👤 {"Select a user to get personalized recommendations:" if language == "English"
    else "Chọn người dùng để nhận gợi ý cá nhân hóa:"}
</div>
""",
unsafe_allow_html=True)
#
# Dropdown chọn mã khách hàng
memb_options = [(row['Member_number'], row['Category']) for index, row in st.session_state.random_trans.iterrows()]

selected_member_tuple = st.selectbox(
    "",
    options=memb_options,
    format_func=lambda x: x[0]  # Hiển thị mã khách hàng
)
st.write("Bạn đã chọn:", selected_member_tuple)

if selected_member_tuple:
    selected_member_number = selected_member_tuple[0]  # Lấy Member_number từ tuple
    st.write("Member: ", selected_member_number)

    # Tìm thông tin phân hạng khách hàng
    segment_info = lookup_member(df_now, selected_member_number)

    st.markdown(
        f"""
        <div style='font-size: 1.25rem; font-weight: 600; margin-bottom: 0rem;'>
             🧠 {"Customer Grouping/Segmentation/Clustering:" if language == "English"
             else "Phân hạng khách hàng: "}
        </div>
        """,
        unsafe_allow_html=True)

    if isinstance(segment_info, pd.DataFrame):
        cluster_name = segment_info['cluster_names'].values[0]
        st.write(cluster_name)

        # 🖼️ Hiển thị ảnh dựa trên tên phân khúc
        image_dict = {
            "Khách hàng trung thành": "images/trung_thanh.png",
            "Khách hàng đã mất": "images/lost.jpg",
            "Khách hàng VIP": "images/VIP.jpg",
            "Khách hàng có nguy cơ rời bỏ":"images/leave.png",
            "Khách hàng không hoạt động":"images/inactive.jpg",
            "Khách hàng cao cấp": "images/VIP2.jpg",
            "Khách hàng tiềm năng": "images/tiemnang2.jpg",
            "Khách hàng trung bình":"images/trungbinh2.png",
            "Khách hàng cần kích hoạt":"images/kichhoat2.jpg"
            # Bạn có thể thêm nhiều phân khúc khác tại đây
        }

        # Hiển thị ảnh nếu có
        image_path = image_dict.get(cluster_name)
        if image_path:
            #st.image(image_path, use_container_width=True)
            st.image(image_path, width=250)
        else:
            st.write("Không có ảnh cho phân khúc này.")

        st.markdown(
            f"""
            <div style='font-size: 1.25rem; font-weight: 600; margin-bottom: 0rem;'>
                🔍 {"The product category the user buys most often:" if language == "English"
                else "Ngành hàng mà người dùng thường mua nhất: "}
            </div>
            """,
            unsafe_allow_html=True)
        st.write(most_frequent_category['Category'].values[0])
    else:
        st.write(segment_info)  # Hiển thị thông báo nếu không tìm thấy
else:
    st.write("Vui lòng chọn một mã khách hàng.")


# (Phần code gợi ý sản phẩm giữ nguyên vì nó không liên quan đến lỗi này)

