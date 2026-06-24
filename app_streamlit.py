import streamlit as st
import json
import boto3

# Konfigurasi AWS SageMaker
ENDPOINT_NAME = "credit-score-endpoint"
REGION_NAME = "us-east-1"

sm_client = boto3.client("sagemaker-runtime", region_name=REGION_NAME)

# Data Kategori
_KNOWN_CAT_VALUES = {
    "Month": [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ],
    "Occupation": [
        "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
        "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager",
        "Musician", "Nurse", "Scientist", "Teacher", "Writer"
    ],
    "Credit_Mix": ["Bad", "Good", "Standard"],
    "Payment_of_Min_Amount": ["No", "Yes"],
    "Payment_Behaviour": [
        "High_spent_Large_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
    ],
}

# Fungsi Inferensi ke SageMaker
def predict_credit_score(data):
    """Mengirim payload JSON ke SageMaker Endpoint untuk mendapatkan prediksi."""
    payload = json.dumps(data)
    
    response = sm_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload
    )
    
    result = json.loads(response["Body"].read().decode("utf-8"))
    return result

# Streamlit UI
def main():
    st.set_page_config(page_title="Credit Scoring System", layout="wide")
    
    st.info(f"Terhubung ke SageMaker Endpoint: `{ENDPOINT_NAME}`")

    st.title("Credit Scoring System")
    st.markdown("Sistem prediksi kelayakan kredit menggunakan Machine Learning di AWS SageMaker.")

    # Sidebar Input
    st.sidebar.header("Data Nasabah")

    st.sidebar.subheader("Informasi Pribadi")
    month      = st.sidebar.selectbox("Bulan", _KNOWN_CAT_VALUES["Month"])
    age        = st.sidebar.number_input("Usia", min_value=18, max_value=100, value=35)
    occupation = st.sidebar.selectbox("Pekerjaan", sorted(_KNOWN_CAT_VALUES["Occupation"]))

    st.sidebar.subheader("Profil Kredit")
    credit_mix       = st.sidebar.selectbox("Credit Mix", _KNOWN_CAT_VALUES["Credit_Mix"])
    payment_min      = st.sidebar.radio("Bayar Minimum Amount?", _KNOWN_CAT_VALUES["Payment_of_Min_Amount"])
    payment_behaviour = st.sidebar.selectbox("Payment Behaviour", _KNOWN_CAT_VALUES["Payment_Behaviour"])

    st.sidebar.subheader("Riwayat Pembayaran")
    delay_due   = st.sidebar.slider("Keterlambatan dari Due Date (hari)", 0, 65, 5)
    num_delayed = st.sidebar.slider("Jumlah Pembayaran Terlambat", 0, 27, 2)
    changed_lim = st.sidebar.number_input("Changed Credit Limit", min_value=0.0, value=3.0, step=0.5)
    num_inq     = st.sidebar.slider("Jumlah Credit Inquiries", 0, 17, 3)
    total_emi   = st.sidebar.number_input("Total EMI per bulan ($)", min_value=0.0, value=200.0, step=10.0)
    amount_inv  = st.sidebar.number_input("Amount Invested Monthly ($)", min_value=0.0, value=300.0, step=50.0)

    st.sidebar.subheader("Riwayat Kredit")
    ch_years  = st.sidebar.slider("Riwayat Kredit (Tahun)", 0, 30, 5)
    ch_months = st.sidebar.slider("Riwayat Kredit (Bulan tambahan)", 0, 11, 3)
    credit_history_age_str = f"{ch_years} Years and {ch_months} Months"

    st.sidebar.subheader("Jenis Pinjaman")
    loan_types_all = [
        "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
        "Home Equity Loan", "Mortgage Loan", "Payday Loan",
        "Personal Loan", "Student Loan",
    ]
    selected_loans   = st.sidebar.multiselect("Pilih yang dimiliki", loan_types_all, default=["Personal Loan"])
    type_of_loan_str = ", ".join(selected_loans) if selected_loans else ""

    # Main Form
    with st.form("input_form"):
        st.write("### Input Data Keuangan")
        col1, col2 = st.columns(2)

        with col1:
            annual_income    = st.number_input("Annual Income ($)", min_value=0, value=50_000, step=1_000)
            monthly_salary   = st.number_input("Monthly Inhand Salary ($)", min_value=0, value=4_000, step=100)
            outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1_000.0, step=100.0)
            monthly_balance  = st.number_input("Monthly Balance ($)", min_value=-10_000.0, value=500.0, step=50.0)

        with col2:
            num_bank_acc    = st.number_input("Jumlah Rekening Bank", min_value=1, max_value=15, value=3)
            num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=11, value=2)
            num_loan        = st.number_input("Jumlah Pinjaman", min_value=0, max_value=9, value=1)
            interest_rate   = st.slider("Interest Rate (%)", 1, 34, 10)
            credit_util     = st.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 30.0)

        submitted = st.form_submit_button("Analisis Kredit", type="primary")

    # Execution
    if submitted:
        input_data = {
            "Month":                    month,
            "Age":                      age,
            "Occupation":               occupation,
            "Annual_Income":            annual_income,
            "Monthly_Inhand_Salary":    monthly_salary,
            "Num_Bank_Accounts":        num_bank_acc,
            "Num_Credit_Card":          num_credit_card,
            "Interest_Rate":            interest_rate,
            "Num_of_Loan":              num_loan,
            "Delay_from_due_date":      delay_due,
            "Num_of_Delayed_Payment":   num_delayed,
            "Changed_Credit_Limit":     changed_lim,
            "Num_Credit_Inquiries":     num_inq,
            "Credit_Mix":               credit_mix,
            "Outstanding_Debt":         outstanding_debt,
            "Credit_Utilization_Ratio": credit_util,
            "Credit_History_Age":       credit_history_age_str,
            "Payment_of_Min_Amount":    payment_min,
            "Total_EMI_per_month":      total_emi,
            "Amount_invested_monthly":  amount_inv,
            "Payment_Behaviour":        payment_behaviour,
            "Monthly_Balance":          monthly_balance,
            "Type_of_Loan":             type_of_loan_str,
        }

        with st.spinner("Menganalisis data via SageMaker Endpoint..."):
            try:
                raw = predict_credit_score(input_data)
                
                # Handle both formats: direct dict atau wrapped dalam "predictions"
                if "predictions" in raw:
                    result = raw["predictions"][0] if isinstance(raw["predictions"], list) else raw["predictions"]
                else:
                    result = raw
                
                label = result.get("predicted_class", "Unknown")
                conf  = result.get("confidence", 0.0)
                probs = result.get("probabilities", {})

                st.divider()
                st.subheader("Hasil Analisis Kredit")

                if label == "Good":
                    st.success(f"**Hasil: {label}**")
                elif label == "Standard":
                    st.warning(f"**Hasil: {label}**")
                else:
                    st.error(f"**Hasil: {label}**")

                st.metric("Confidence Score", f"{conf:.2f}%")

                st.markdown("##### Distribusi Probabilitas per Kelas")
                c1, c2, c3 = st.columns(3)
                c1.metric("Good",     f"{probs.get('Good',     0):.1f}%")
                c2.metric("Standard", f"{probs.get('Standard', 0):.1f}%")
                c3.metric("Poor",     f"{probs.get('Poor',     0):.1f}%")

            except Exception as e:
                st.error("Gagal terhubung ke SageMaker Endpoint.")
                st.error(f"Detail Error: {e}")

if __name__ == "__main__":
    main()
