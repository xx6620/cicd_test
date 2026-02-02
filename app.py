import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# 1. 페이지 전체 설정
st.set_page_config(page_title="로아 아이템 시세 예측", layout="wide")

# 2. 커스텀 CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4562; }
    </style>
    """, unsafe_allow_html=True)

# 3. 사이드바
with st.sidebar:
    st.title("분석")
    st.info("분석 대상을 설정해주세요.")
    
    # 아이템 검색
    item_search = st.text_input("아이템 이름 검색", placeholder="예: 원한, 명파")
    
    # 예측 설정
    st.divider()
    predict_range = st.select_slider(
        "예측 기간 설정",
        options=["24시간", "48시간", "수요일 리셋까지"],
        value="24시간"
    )
    
    # 앙상블 비중 조절
    st.subheader("모델 가중치 설정")
    w_lstm = st.slider("LSTM", 0.0, 1.0, 0.34)
    w_ml = st.slider("ML (XGB/LGBM)", 0.0, 1.0, 0.33)
    w_prophet = st.slider("NeuralProphet", 0.0, 1.0, 0.33)

# 4. 메인 대시보드
st.title("로스트아크 시세 예측")

# 상단 주요 지표 (Metrics)
m1, m2, m3, m4 = st.columns(4)
m1.metric("현재 시세", "32,450 G", "+1.2%")
m2.metric("24h 최저가", "31,200 G", "-2.5%")
m3.metric("예측 최종가", "33,800 G", "상승세", delta_color="normal")
m4.metric("AI 임팩트 점수", "0.85", "강한 호재")

st.divider()

# 메인 차트 및 리포트 영역
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📊 통합 앙상블 예측 경로")
    # 차트 들어갈 자리 (Placeholder)
    st.image("https://via.placeholder.com/800x400.png?text=Main+Ensemble+Chart+Placeholder")
    
    # 상세 탭 (개별 모델 확인)
    tab1, tab2, tab3 = st.tabs(["LSTM/ML 추세", "NeuralProphet", "보조지표"])
    with tab1:
        st.write("모델별 상세 데이터가 출력됩니다.")
    with tab2:
        st.write("수요일 주기성이 반영된 프로핏 차트가 출력됩니다.")
    with tab3:
        st.write("RSI, 볼린저 밴드 현황입니다.")

with right_col:
    st.subheader("🤖 AI 전략 리포트")
    with st.expander("📍 현재 상황 요약", expanded=True):
        st.write("공지사항(13326) 분석 결과, 보석 보상 증가로 인한...")
    
    with st.expander("🛡️ 대응 가이드"):
        st.warning("경로 내 최저가 구간이 존재합니다. 지금 즉시 매수하지 마세요.")
    
    st.success("결론: 이번 주말까지는 홀딩 후 수요일 직전 매도 추천")

# 하단 로그 창
with st.expander("📝 시스템 로그"):
    st.code("Loading model... [OK]\nData Syncing... [OK]\nEnsemble weight applied: 0.34:0.33:0.33")