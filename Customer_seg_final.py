import streamlit as st
import pandas as pd
import pickle
import numpy as np
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import StandardScaler  

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

def lookup_member_rfm(df, member_number):
    result = df[df['Member_number'].astype(str) == str(member_number)]
    if result.empty:
        return f"Không tìm thấy thành viên với Member_number = {member_number}"
    return result[['Recency', 'Frequency', 'Monetary']]

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

import streamlit as st
import pandas as pd

# Bước 1: Giao diện tiêu đề
st.title("Phân khúc Khách hàng")

# Khởi tạo session state nếu cần
if 'df_trans' not in st.session_state:
    st.session_state['df_trans'] = None
if 'random_trans' not in st.session_state:
    st.session_state['random_trans'] = None

# Bước 1: Chọn dataset
st.markdown("## 📁 Bước 1: Chọn nguồn dữ liệu")

# Chia thành 2 cột: chọn dataset có sẵn hoặc tải lên
#col1, col2 = st.columns(2)
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 🗂️ Dataset có sẵn")
    if st.button("📌 Sử dụng dataset mẫu"):
        st.session_state['df_trans'] = pd.read_csv('data_segmentation_total.csv')
        st.session_state['dataset_source'] = "default"
        st.success("Đã tải dữ liệu mẫu thành công!")

with col2:
    st.markdown("### ⬆️ Tải lên dataset của bạn")
    uploaded_file = st.file_uploader("Tải lên file CSV", type="csv")
    if uploaded_file is not None:
        try:
            st.session_state['df_trans'] = pd.read_csv(uploaded_file, encoding='latin-1')
            st.session_state['dataset_source'] = "uploaded"
            st.success("Tải file thành công!")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi đọc file: {e}")

# Hiển thị thông báo trạng thái nguồn dữ liệu đã chọn
if 'dataset_source' in st.session_state:
    source = st.session_state['dataset_source']
    if source == "default":
        st.info("📊 Đang sử dụng **dataset có sẵn**.")
    elif source == "uploaded":
        st.info("📥 Đang sử dụng **dataset bạn đã tải lên**.")


# Bước 2: Hiển thị và xử lý dữ liệu nếu đã chọn
if st.session_state['df_trans'] is not None:
    df_trans = st.session_state['df_trans']

    # Kiểm tra cột cần thiết
    required_cols = ['price', 'items', 'Date']
    if all(col in df_trans.columns for col in required_cols):
        # Feature Engineering
        df_trans['Gross'] = df_trans['price'] * df_trans['items']
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

    
    else:
        st.warning(f"Dataset thiếu một số cột cần thiết: {', '.join([col for col in required_cols if col not in df_trans.columns])}")

    # Hiển thị ngẫu nhiên 15 dòng
    st.subheader("Dataset")
    if st.session_state['dataset_source'] == "default":
        st.info("Đang hiển thị ngẫu nhiên 15 khách hàng từ dataset có sẵn.")
    elif st.session_state['dataset_source'] == "uploaded":
        st.info("Đang hiển thị ngẫu nhiên 15 mẫu từ file đã tải lên.")

    st.session_state['random_trans'] = df_trans.sample(n=15)
    st.subheader("15 Mẫu Khách Hàng Ngẫu Nhiên")
    st.dataframe(st.session_state['random_trans'])

    # Bước 3: Chọn mã khách hàng
    st.subheader("Bước 2: Chọn cách nhập Mã Khách Hàng")

    # Radio chọn cách nhập thông tin
    method = st.radio(
        "Chọn cách nhập thông tin khách hàng",
        options=["Chọn từ danh sách", "Nhập mã khách hàng"],
        index=0  # Mặc định chọn "Chọn từ danh sách"
    )

    if method == "Chọn từ danh sách":
        st.write("🔽 Chọn Mã Khách Hàng từ danh sách:")
        memb_options = [
            (row['Member_number'], row['Category']) 
            for index, row in st.session_state.random_trans.iterrows()
        ]

        selected_member_tuple = st.selectbox(
            "Danh sách mã khách hàng:",
            options=memb_options,
            format_func=lambda x: f"{x[0]} - {x[1]}"
        )
        st.write("✅ Bạn đã chọn Mã Khách Hàng:", selected_member_tuple[0])

    elif method == "Nhập mã khách hàng":
        selected_member_tuple = st.text_input("Nhập Mã Khách Hàng:")
        if selected_member_tuple:
            st.write(f"✅ Bạn đã nhập Mã Khách Hàng: '{selected_member_tuple}'")


    
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
    
    # GỌI Model KMEANS hoặc Manual LÊN DÙNG
    df_now = Manual_segments(data_RFM, recency_col='Recency', frequency_col='Frequency', monetary_col='Monetary') if (filtering_method == "🧠 Manual RFM" or filtering_method == "🧠 Phân khúc theo kinh nghiệm ") else segment_customers_kmeans(data_RFM)

    if selected_member_tuple:
        if method == "Chọn từ danh sách":
            selected_member_number = selected_member_tuple[0]  # Lấy Member_number từ tuple
        else:selected_member_number = selected_member_tuple
        st.write("Member: ", selected_member_number)   
        st.write('Thông tin khách hàng:',lookup_member_rfm(df_now,selected_member_number))

        # Tìm thông tin phân hạng khách hàng

        if method == "Chọn từ danh sách":
            segment_info = lookup_member(df_now, selected_member_tuple[0])
        else: segment_info = lookup_member(df_now, selected_member_tuple)    

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
                "Khách hàng trung bình":"images/trungbinh2.jpg",
                "Khách hàng cần kích hoạt":"images/kichhoat2.jpg"
                # Bạn có thể thêm nhiều phân khúc khác tại đây
            }
        # 💡 Dictionary gợi ý chiến lược theo phân khúc
            advice_dict = {
                "Khách hàng trung thành": "🎁 Tăng giá trị chi tiêu và duy trì lòng trung thành:Chương trình tích điểm đổi quà, email nhắc duy trì tương tác, khuyến mãi cá nhân hóa.",
                "Khách hàng đã mất": "📣 Hãy khuyến mãi 1234 để thu hút họ quay lại.",
                "Khách hàng VIP": "💎Giữ chân và tối ưu hóa giá trị:Cung cấp quyền lợi đặc biệt như giao hàng miễn phí, ưu tiên đặt hàng, hoặc sản phẩm độc quyền, Upsell/Cross-sell, Chương trình tri ân ",
                "Khách hàng có nguy cơ rời bỏ": "⚠️ : Ngăn chặn rời bỏ và kích thích mua hàng: Gửi mã giảm giá có thời hạn ngắn, tương tác đa kênh để nhắc nhở họ về thương hiệu",
                "Khách hàng không hoạt động": "⏸️ Tái kích hoạt hoặc chấp nhận mất khách: thử các ưu đãi khác biệt (ví dụ: tặng sản phẩm miễn phí khi mua hàng).",
                "Khách hàng cao cấp": "🏆 Chi tiêu cao, mua hàng thường xuyên, đóng góp lớn vào doanh thu. Hãy khuyến mãi để làm họ cảm thấy đặc biệt.",
                "Khách hàng tiềm năng": "🚀 cần được khuyến khích mua thường xuyên hơn: o	Khuyến mãi tăng tần suất: Áp dụng ưu đãi Mua 5 lần trong 30 ngày, nhận voucher 10 USD để thúc đẩy Frequency,Gói combo sản phẩm ",
                "Khách hàng trung bình": "📦 o	Ưu đãi theo ngưỡng chi tiêu: Đưa ra khuyến mãi như Chi tiêu trên 60 USD giảm 10 USD để tăng Monetary, Chương trình giới thiệu mở rộng tệp khách hàng ",
                "Khách hàng cần kích hoạt": "🔔Nhóm này có nguy cơ rời bỏ cao do Recency quá dài , gửi email hoặc tin nhắn với ưu đãi mạnh, khuyến mãi theo mùa, khảo sát nhu cầu ."
            }
            # Hiển thị ảnh nếu có
            image_path = image_dict.get(cluster_name)
            if image_path:
                #st.image(image_path, use_container_width=True)
                st.image(image_path, width=250)
            else:
                st.write("Không có ảnh cho phân khúc này.")
            # Hiển thị lời khuyên
            advice = advice_dict.get(cluster_name)
            if advice:
                st.markdown(f"**Lời khuyên:** {advice}")
            else:
                st.info("Chưa có lời khuyên cho phân khúc này.")
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

else:
   
    st.info("Vui lòng chọn dataset bằng cách nhấn nút hoặc tải lên file CSV.")




