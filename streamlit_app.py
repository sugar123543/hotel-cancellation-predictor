
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="호텔 예약 취소 예측",
    page_icon="🏨",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load(
        "hotel_cancellation_model_compressed.pkl"
    )

model = load_model()

st.title("🏨 호텔 예약 취소 예측 시스템")
st.write("머신러닝 기반 호텔 예약 취소 위험 분석")
st.info("예약 정보를 입력하고 예측 버튼을 눌러주세요.")

with st.form("reservation_form"):
    st.subheader("예약 정보 입력")

    adults = st.number_input("성인 수", 0, 4, 2)
    children = st.number_input("어린이 수", 0, 10, 0)
    weekend = st.number_input("주말 숙박 일수", 0, 20, 1)
    weekday = st.number_input("평일 숙박 일수", 0, 30, 2)

    meal = st.selectbox(
        "식사 플랜",
        ["Meal Plan 1", "Meal Plan 2",
         "Meal Plan 3", "Not Selected"]
    )

    parking = st.selectbox("주차 공간 요청", [0, 1])

    room = st.selectbox(
        "객실 유형",
        [f"Room_Type {i}" for i in range(1, 8)]
    )

    lead = st.number_input(
        "예약 리드타임 (일)", 0, 500, 60
    )

    year = st.selectbox("체크인 연도", [2017, 2018])
    month = st.slider("체크인 월", 1, 12, 6)
    day = st.slider("체크인 일", 1, 31, 15)

    market = st.selectbox(
        "예약 채널",
        ["Online", "Offline", "Corporate",
         "Complementary", "Aviation"]
    )

    repeat = st.selectbox("재방문 고객 여부", [0, 1])

    prev_cancel = st.number_input(
        "과거 취소 횟수", 0, 20, 0
    )

    prev_not_cancel = st.number_input(
        "과거 예약 유지 횟수", 0, 100, 0
    )

    price = st.number_input(
        "평균 객실 가격", 0.0, 1000.0, 100.0
    )

    requests = st.number_input(
        "특별 요청 수", 0, 5, 0
    )

    submitted = st.form_submit_button(
        "🔍 예약 취소 확률 예측",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "no_of_adults": adults,
        "no_of_children": children,
        "no_of_weekend_nights": weekend,
        "no_of_week_nights": weekday,
        "type_of_meal_plan": meal,
        "required_car_parking_space": parking,
        "room_type_reserved": room,
        "lead_time": lead,
        "arrival_year": year,
        "arrival_month": month,
        "arrival_date": day,
        "market_segment_type": market,
        "repeated_guest": repeat,
        "no_of_previous_cancellations": prev_cancel,
        "no_of_previous_bookings_not_canceled": prev_not_cancel,
        "avg_price_per_room": price,
        "no_of_special_requests": requests
    }])

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 예측 결과")
    st.metric("예약 취소 예상 확률",
              f"{probability * 100:.1f}%")
    st.progress(float(probability))

    if probability >= 0.7:
        st.error("🔴 취소 고위험군")
        st.write("예약 재확인 및 사전 결제 혜택 제안 권장")
    elif probability >= 0.4:
        st.warning("🟡 취소 중위험군")
        st.write("체크인 전 예약 확인 메시지 권장")
    else:
        st.success("🟢 취소 저위험군")
        st.write("일반 예약 관리 및 부가서비스 제안 권장")

    st.caption(
        "본 결과는 학습 데이터 기반 예측이며 "
        "실제 취소 여부를 보장하지 않습니다."
    )
