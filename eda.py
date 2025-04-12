import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt
import plotly.express as px

# Set language from session
language = st.session_state.get("language", "English")

dataset_name = "data_segmentation_total.csv"
# === 🧹 Load the Data ===
@st.cache_data
def load_data():
    data = pd.read_csv(dataset_name, sep=',')
    
    return data

df = load_data()

# === 🧾 Dataset Summary ===
if language == "English":
    st.subheader(f"📊 Dataset Summary")
    st.write("Shape:", df.shape)
    st.write("Sample Data:")
    text = "Select the number of samples to display"

else:
    st.subheader(f"📊 Thống kê Dữ liệu")
    st.write("Kích thước:", df.shape)
    st.write("Dữ liệu mẫu:")
    text = "Chọn số lượng mẫu dữ liệu để hiển thị"
    
value = value = st.slider(text, min_value=0, max_value=100, value=5)
st.dataframe(df.head(value))
# === 🛍️ Top Categories ===
st.write("#### Top Categories" if language == "English" else "#### Danh mục phổ biến")
value = value = st.slider("Select the number of category to display" if language == "English" else "Chọn số lượng thể loại để hiển thị", min_value=0, max_value=df["Category"].nunique(), value=5)
top_categories = df['Category'].value_counts().head(value)
st.bar_chart(top_categories)

st.markdown("---")

# === 🛍️ Number of Items ===
st.write("#### Number of Items" if language == "English" else "#### Số lượng sản phẩm")

st.write("Total number of items: " if language == "English" else "Tổng số sản phẩm", df["items"].count())
items = df['items'].value_counts()
st.bar_chart(items, horizontal = True)

st.markdown("---")

# === ☁️ WordCloud for Product Names ===
if language == "English":
    st.write("#### ☁️ Word Cloud of Product Names")
else:
    st.write("#### ☁️ Word Cloud từ của Tên sản phẩm")

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['productName'].astype(str).tolist()))
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)
        
st.markdown("---")

# PRICE
if language == "English":
    st.write(f"#### 💹💹💹 Price Distribution")
else:
    st.write(f"#### 💹💹💹 Phân phối giá cả")

col1, col2, col3 = st.columns(3)

col1.metric("⭐ Max" if language == "English" else "⭐ Cao nhất",  f"{df['price'].max():.2f}".replace(",", ".") + " $")

col2.metric("🔻 Min" if language == "English" else "🔻 Thấp nhất", f"{df['price'].min():.2f}".replace(",", ".") + " $")

col3.metric("📊 Mean" if language == "English" else "📊 Trung bình", f"{df['price'].mean():.2f}".replace(",", ".") + " $")

fig2 = px.box(
            df,
            x='price',
            points='outliers',  # Only show outliers
            hover_data=["productName", "Category", "price"],
            title=" ",
)
fig2.update_traces(
            marker=dict(
                color="#1f77b4",  # deep blue
                size=8,
                line=dict(width=1.5, color='#ffffff')  # white border
            )
        )
fig2.update_layout(
            xaxis=dict(
                title= " ",
                title_font=dict(size=20, color="#1f77b4"),
                tickfont=dict(size=16, color="#1f77b4"),
                showgrid=True,
                gridcolor="rgba(173, 216, 230, 0.3)",  # light blue grid
                linecolor="#1f77b4",
                zerolinecolor="#1f77b4"
            ),
            yaxis=dict(
                title=None,
                tickfont=dict(size=16, color="#1f77b4"),
                showgrid=True,
                gridcolor="rgba(173, 216, 230, 0.3)",
                linecolor="#1f77b4",
                zerolinecolor="#1f77b4"
            ),
            plot_bgcolor='rgba(240, 248, 255, 0.6)',  # light blueish background
            paper_bgcolor='rgba(240, 248, 255, 1)',   # light paper color
            font=dict(family="Roboto, sans-serif", size=18, color="#1f77b4"),
            margin=dict(l=40, r=40, t=70, b=40),
            height=500
        )


st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
# === 👥 User Insights ===
if language == "English":
    st.write("#### 👥 User Activity")
else:
    st.write("#### 👥 Hoạt động người dùng")
user_counts = df['Member_number'].value_counts().reset_index()
user_counts.columns = ['Member_number', 'Bought_Products']

st.write("*Total number of users*:" if language == "English" else "*Tổng số người dùng*", user_counts.shape[0])

st.dataframe(user_counts)
st.markdown("---")

# === ⭐ Date analysis ===
if language == "English":
    st.write(f"#### ⭐⭐⭐ Transaction Date - Time Series Analysis")
else:
    st.write(f"#### ⭐⭐⭐ Phân tích giao dịch theo ngày tháng - Time Series")
df['transaction_date'] = pd.to_datetime(df['Date'])

min_date = df['transaction_date'].min()
max_date = df['transaction_date'].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Filter based on selected range
if len(date_range) == 2:
    start_date, end_date = date_range
    days_selected = (end_date - start_date).days
    df = df[(df['transaction_date'] >= pd.to_datetime(start_date)) & (df['transaction_date'] <= pd.to_datetime(end_date))]
    
df['year'] = df['transaction_date'].dt.year
df['month'] = df['transaction_date'].dt.month
df['day'] = df['transaction_date'].dt.day
df['weekday'] = df['transaction_date'].dt.day_name()
df['week'] = df['transaction_date'].dt.isocalendar().week

daily_sales = df.groupby('transaction_date').size()

st.subheader("Daily Transactions")
st.line_chart(daily_sales)

monthly_sales = df.groupby(df['transaction_date'].dt.to_period("M")).size()
st.subheader("Monthly Transactions")
st.bar_chart(monthly_sales)

monthly_sales = df.groupby(['year', 'month']).size().reset_index(name='count')
monthly_sales['month_year'] = monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str).str.zfill(2)
st.subheader("Monthly Transactions Fix")
st.bar_chart(monthly_sales.set_index('month_year')['count'])

weekday_sales = df['weekday'].value_counts().sort_index()
st.subheader("Transactions by Weekday")
st.bar_chart(weekday_sales)

from statsmodels.tsa.seasonal import seasonal_decompose
if (days_selected >= 14):
    daily_series = daily_sales.asfreq('D').fillna(0)
    result = seasonal_decompose(daily_series, model='additive')

    st.subheader("Seasonal Decomposition of Time Series")

    st.caption("📈 Trend")
    st.line_chart(result.trend)

    st.caption("🔁 Seasonal Pattern")
    st.line_chart(result.seasonal)

    st.caption("📉 Residuals (Noise)")
    st.line_chart(result.resid)

else: 
    st.warning("⚠️ Please choose a date range of at least 14 days for time series analysis.")

# # === 📥 Download Option ===
if language == "English":
    st.sidebar.download_button(f"📥 Download dataset", df.to_csv(index=False))
else:
    st.sidebar.download_button(f"📥 Tải xuống dữ liệu", df.to_csv(index=False))
