import streamlit as st
import pandas as pd
import numpy as np
import datetime


st.set_page_config(page_title="quant - 예측")


st.title("가상 자산 시세 예측")
st.subheader("결과")

st.markdown(f"""
현재 시간: **{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**
""")


with st.sidebar:
    st.header("설정")
    item_name = st.text_input("아이템 이름 입력", value="강화석")
    predict_days = st.slider("예측 기간 (일)", 1, 7, 3)
    st.info("사이드바에서 변수를 조절할 수 있습니다.")


chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['현재 시세', '예측 시세(LSTM)', '예측 시세(XGBoost)']
)


st.divider()
st.write(f"{item_name} 가격 변동 및 예측 추이")
st.line_chart(chart_data)


st.write("#최근 수집 데이터 (Raw Data)")
st.dataframe(chart_data.tail(10), use_container_width=True)


st.success("DB 연결 준비")