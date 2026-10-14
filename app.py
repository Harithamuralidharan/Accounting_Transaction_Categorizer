import streamlit as st
import pandas as pd
import pickle
import os
import re

from io import BytesIO


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="AI Accounting Transaction Categorizer",

    page_icon="💰",

    layout="wide"

)


MODEL_FILE = (
    "Models/transaction_model.pkl"
)

VENDOR_FILE = (
    "Data/vendor_master.csv"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(
        MODEL_FILE
    ):

        st.error(
            "ML model not found. "
            "Please run train_model.py first."
        )

        st.stop()

    with open(

        MODEL_FILE,

        "rb"

    ) as file:

        model = pickle.load(
            file
        )

    return model


model = load_model()


# ============================================================
# VENDOR MASTER - LOAD
# ============================================================

def load_vendor_master():

    os.makedirs(

        "Data",

        exist_ok=True

    )


    if not os.path.exists(
        VENDOR_FILE
    ):

        df = pd.DataFrame(

            columns=[

                "Vendor Name",

                "Keywords"

            ]

        )

        df.to_csv(

            VENDOR_FILE,

            index=False

        )

        return df


    df = pd.read_csv(

        VENDOR_FILE

    )


    if "Vendor Name" not in df.columns:

        df["Vendor Name"] = ""


    if "Keywords" not in df.columns:

        df["Keywords"] = ""


    return df


# ============================================================
# SAVE / UPDATE VENDOR
# ============================================================

def save_vendor(

    vendor_name,

    keywords

):

    vendor_name = (
        vendor_name.strip()
    )

    keywords = (
        keywords.strip()
    )


    if not vendor_name or not keywords:

        return False


    vendor_df = (
        load_vendor_master()
    )


    existing = vendor_df[

        vendor_df[
            "Vendor Name"
        ]
        .astype(str)
        .str.lower()
        ==
        vendor_name.lower()

    ]


    if len(existing) > 0:

        index = existing.index[0]

        vendor_df.loc[

            index,

            "Keywords"

        ] = keywords


    else:

        new_row = pd.DataFrame({

            "Vendor Name": [

                vendor_name

            ],

            "Keywords": [

                keywords

            ]

        })


        vendor_df = pd.concat(

            [

                vendor_df,

                new_row

            ],

            ignore_index=True

        )


    vendor_df.to_csv(

        VENDOR_FILE,

        index=False

    )


    return True


# ============================================================
# FIND VENDOR FROM VENDOR MASTER
# ============================================================

def find_vendor(

    description

):

    text = (

        str(description)
        .lower()
        .strip()

    )


    vendor_df = (
        load_vendor_master()
    )


    for _, row in vendor_df.iterrows():

        vendor_name = (

            str(
                row["Vendor Name"]
            )
            .strip()

        )


        keywords = (

            str(
                row["Keywords"]
            )
            .split("|")

        )


        for keyword in keywords:

            keyword = (
                keyword.strip()
                .lower()
            )


            if (

                keyword
                and keyword in text

            ):

                return vendor_name


    return None


# ============================================================
# FALLBACK VENDOR EXTRACTION
# ============================================================

def extract_vendor_from_text(

    description

):

    text = (
        str(description)
        .strip()
    )


    patterns = [

        r"paid to ([a-zA-Z0-9 &.\-]+)",

        r"payment to ([a-zA-Z0-9 &.\-]+)",

        r"received from ([a-zA-Z0-9 &.\-]+)",

        r"paid ([a-zA-Z0-9 &.\-]+) for",

        r"payment from ([a-zA-Z0-9 &.\-]+)"

    ]


    for pattern in patterns:

        match = re.search(

            pattern,

            text,

            re.IGNORECASE

        )


        if match:

            vendor = (
                match.group(1)
                .strip()
            )


            vendor = re.sub(

                r"\b(the|for|bill|invoice|amount)\b",

                "",

                vendor,

                flags=re.IGNORECASE

            ).strip()


            if vendor:

                return vendor.title()


    return "Unknown Vendor"


# ============================================================
# FINAL VENDOR DETECTION
# ============================================================

def get_vendor(

    description

):

    vendor = find_vendor(

        description

    )


    if vendor:

        return vendor


    return extract_vendor_from_text(

        description

    )


# ============================================================
# CATEGORY PREDICTION
# ============================================================

def predict_category(

    description

):

    description = (
        str(description)
        .strip()
    )


    if not description:

        return (
            "Uncategorized",
            0.0
        )


    prediction = model.predict(

        [description]

    )[0]


    probabilities = model.predict_proba(

        [description]

    )


    confidence = (

        probabilities.max()
        * 100

    )


    return (

        prediction,

        confidence

    )


# ============================================================
# REVIEW STATUS
# ============================================================

def get_review_status(

    confidence

):

    if confidence >= 80:

        return "High Confidence"

    elif confidence >= 60:

        return "Review Recommended"

    else:

        return "Needs Review"


# ============================================================
# FIND DESCRIPTION COLUMN
# ============================================================

def find_description_column(

    df

):

    df.columns = (

        df.columns
        .astype(str)
        .str.strip()

    )


    possible_columns = [

        "Vendor Name / Description",

        "Memo / Description",

        "Memo (or) Description",

        "Memo",

        "Description",

        "description",

        "DESCRIPTION",

        "transaction_description",

        "Transaction Description",

        "Narration",

        "Details",

        "Particulars"

    ]


    for column in possible_columns:

        if column in df.columns:

            return column


    return None


# ============================================================
# FIND OPTIONAL COLUMN
# ============================================================

def find_column(

    df,

    possible_columns

):

    for column in possible_columns:

        if column in df.columns:

            return column


    return None


# ============================================================
# PROCESS SINGLE TRANSACTION
# ============================================================

def create_transaction_result(

    date,

    description,

    amount,

    debit_credit=None

):

    vendor = get_vendor(

        description

    )


    category, confidence = (

        predict_category(

            description

        )

    )


    status = get_review_status(

        confidence

    )


    result = pd.DataFrame({

        "Date": [

            date

        ],

        "Vendor Name / Description": [

            description

        ],

        "Amount": [

            amount

        ],

        "Debit/Credit": [

            debit_credit

            if debit_credit

            else ""

        ],

        "Vendor Name": [

            vendor

        ],

        "Accounting Category": [

            category

        ],

        "Confidence": [

            round(
                confidence,
                2
            )

        ],

        "Review Status": [

            status

        ]

    })


    return result


# ============================================================
# PROCESS EXCEL
# ============================================================

def process_excel(

    df

):

    # --------------------------------------------------------
    # Find Description
    # --------------------------------------------------------

    description_column = (

        find_description_column(

            df

        )

    )


    if description_column is None:

        return None, (

            "❌ Description column not found.\n\n"

            "Your Excel should contain one of these:\n\n"

            "• Memo / Description\n"

            "• Description\n"

            "• Memo\n"

            "• Narration\n"

            "• Details\n"

            "• Particulars"

        )


    # --------------------------------------------------------
    # Copy input
    # --------------------------------------------------------

    output_df = df.copy()


    # --------------------------------------------------------
    # Find optional columns
    # --------------------------------------------------------

    date_column = find_column(

        output_df,

        [

            "Date",

            "Transaction Date",

            "date",

            "DATE"

        ]

    )


    amount_column = find_column(

        output_df,

        [

            "Amount",

            "amount",

            "AMOUNT",

            "Transaction Amount"

        ]

    )


    debit_credit_column = find_column(

        output_df,

        [

            "Debit/Credit",

            "Debit Credit",

            "DebitCredit",

            "Type",

            "Transaction Type"

        ]

    )


    # --------------------------------------------------------
    # Vendor
    # --------------------------------------------------------

    output_df["Vendor Name"] = (

        output_df[

            description_column

        ]

        .fillna("")

        .astype(str)

        .apply(get_vendor)

    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    categories = []

    confidences = []

    statuses = []


    for description in (

        output_df[

            description_column

        ]

        .fillna("")

        .astype(str)

    ):

        category, confidence = (

            predict_category(

                description

            )

        )


        status = (

            get_review_status(

                confidence

            )


        )


        categories.append(

            category

        )


        confidences.append(

            round(
                confidence,
                2
            )

        )


        statuses.append(

            status

        )


    # --------------------------------------------------------
    # Add results
    # --------------------------------------------------------

    output_df[

        "Accounting Category"

    ] = categories


    output_df[

        "Confidence"

    ] = confidences


    output_df[

        "Review Status"

    ] = statuses


    # --------------------------------------------------------
    # Create final output
    # --------------------------------------------------------

    final_df = pd.DataFrame()


    # Date

    if date_column:

        final_df["Date"] = (

            output_df[
                date_column
            ]

        )

    else:

        final_df["Date"] = ""


    # Description

    final_df[

        "Vendor Name / Description"

    ] = (

        output_df[

            description_column

        ]

    )


    # Amount

    if amount_column:

        final_df["Amount"] = (

            output_df[
                amount_column
            ]

        )

    else:

        final_df["Amount"] = ""




    # Vendor

    final_df[

        "Vendor Name"

    ] = (

        output_df[

            "Vendor Name"

        ]

    )


    # Accounting Category

    final_df[

        "Accounting Category"

    ] = (

        output_df[

            "Accounting Category"

        ]

    )




    return final_df, None


# ============================================================
# CREATE EXCEL
# ============================================================

def create_excel(

    df

):

    output = BytesIO()


    with pd.ExcelWriter(

        output,

        engine="openpyxl"

    ) as writer:

        df.to_excel(

            writer,

            index=False,

            sheet_name="Categorized Transactions"

        )


    output.seek(0)


    return output


# ============================================================
# MAIN TITLE
# ============================================================

st.title(

    "💰 AI Accounting Transaction Categorizer"

)


st.write(

    "Automatically classify financial transactions "
    "into accounting categories using Machine Learning."

)


st.caption(

    "Debit/Credit column is optional. "
    "The model can categorize transactions using description alone."

)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(

    [

        "📊 Excel Upload",

        "✍️ Single Transaction",

        "🏢 Vendor Master"

    ]

)


# ============================================================
# TAB 1 - EXCEL UPLOAD
# ============================================================

with tab1:

    st.subheader(

        "📊 Upload Transaction File"

    )


    st.info(

        "Required: Description / Memo column\n\n"

        "Optional: Date, Amount, Debit/Credit"

    )


    uploaded_file = st.file_uploader(

        "Upload Excel or CSV file",

        type=[

            "xlsx",

            "xls",

            "csv"

        ]

    )


    if uploaded_file:

        try:

            if uploaded_file.name.lower().endswith(

                ".csv"

            ):

                input_df = pd.read_csv(

                    uploaded_file

                )

            else:

                input_df = pd.read_excel(

                    uploaded_file

                )


        except Exception as e:

            st.error(

                f"Unable to read file: {e}"

            )

            st.stop()


        st.subheader(

            "📄 Input Transactions"

        )


        st.dataframe(

            input_df,

            use_container_width=True

        )


        if st.button(

            "🤖 Categorize Transactions",

            type="primary",

            key="categorize_excel"

        ):

            with st.spinner(

                "AI is processing transactions..."

            ):

                result_df, error = (

                    process_excel(

                        input_df

                    )

                )


            if error:

                st.error(

                    error

                )

            else:

                st.success(

                    "✅ Transactions categorized successfully!"

                )


                st.subheader(

                    "📋 Categorized Transactions"

                )


                st.dataframe(

                    result_df,

                    use_container_width=True

                )


                # Summary

                st.subheader(

                    "📊 Summary"

                )


                col1, col2, col3 = st.columns(

                    3

                )


                with col1:

                    st.metric(

                        "Transactions",

                        len(result_df)

                    )


                # Download

                excel_file = create_excel(

                    result_df

                )


                st.download_button(

                    label=(

                        "📥 Download Categorized Excel"

                    ),

                    data=excel_file,

                    file_name=(

                        "categorized_transactions.xlsx"

                    ),

                    mime=(

                        "application/vnd.openxmlformats-officedocument."

                        "spreadsheetml.sheet"

                    )

                )


# ============================================================
# TAB 2 - SINGLE TRANSACTION
# ============================================================

with tab2:

    st.subheader(

        "✍️ Enter Single Transaction"

    )


    date = st.date_input(

        "Date"

    )


    description = st.text_input(

        "Vendor Name / Description",

        placeholder=(

            "Example: Payment to ABC Traders"

        )

    )


    amount = st.number_input(

        "Amount",

        min_value=0.0,

        step=0.01

    )


    debit_credit = st.selectbox(

        "Debit/Credit (Optional)",

        [

            "",

            "Debit",

            "Credit"

        ]

    )


    if st.button(

        "🔍 Categorize Transaction",

        type="primary",

        key="categorize_single"

    ):

        if not description.strip():

            st.warning(

                "Please enter Vendor Name / Description."

            )

        else:

            result = create_transaction_result(

                date,

                description,

                amount,

                debit_credit

            )


            st.success(

                "✅ Transaction categorized successfully!"

            )


            st.dataframe(

                result,

                use_container_width=True

            )


            excel_file = create_excel(

                result

            )


            st.download_button(

                label=(

                    "📥 Download Excel"

                ),

                data=excel_file,

                file_name=(

                    "single_categorized_transaction.xlsx"

                ),

                mime=(

                    "application/vnd.openxmlformats-officedocument."

                    "spreadsheetml.sheet"

                )

            )


# ============================================================
# TAB 3 - VENDOR MASTER
# ============================================================

with tab3:

    st.subheader(

        "🏢 Vendor Master"

    )


    st.write(

        "Add, edit, or delete vendors. "

        "Saved vendors will be remembered "

        "for future transactions."

    )


    # ========================================================
    # ADD VENDOR
    # ========================================================

    st.markdown(

        "### ➕ Add New Vendor"

    )


    new_vendor_name = st.text_input(

        "Vendor Name",

        placeholder="Example: ABC Enterprises",

        key="new_vendor_name"

    )


    new_keywords = st.text_input(

        "Vendor Keywords",

        placeholder=(

            "Example: abc enterprises|abc ent|abc enterprises pvt ltd"

        ),

        key="new_vendor_keywords"

    )


    if st.button(

        "💾 Save Vendor",

        type="primary",

        key="save_vendor"

    ):

        if not new_vendor_name.strip():

            st.warning(

                "Please enter Vendor Name."

            )

        elif not new_keywords.strip():

            st.warning(

                "Please enter at least one keyword."

            )

        else:

            save_vendor(

                new_vendor_name,

                new_keywords

            )


            st.success(

                f"✅ {new_vendor_name} saved successfully!"

            )


            st.rerun()


    # ========================================================
    # SAVED VENDORS
    # ========================================================

    st.markdown("---")

    st.markdown(

        "### 📋 Saved Vendors"

    )


    vendor_df = load_vendor_master()


    if vendor_df.empty:

        st.info(

            "No vendors saved yet."

        )

    else:

        st.dataframe(

            vendor_df,

            use_container_width=True

        )


    # ========================================================
    # EDIT VENDOR
    # ========================================================

    st.markdown("---")

    st.markdown(

        "### ✏️ Edit Vendor"

    )


    if not vendor_df.empty:

        vendor_list = (

            vendor_df[

                "Vendor Name"

            ]

            .tolist()

        )


        selected_vendor = st.selectbox(

            "Select Vendor to Edit",

            vendor_list,

            key="edit_vendor_select"

        )


        selected_row = vendor_df[

            vendor_df[

                "Vendor Name"

            ]

            == selected_vendor

        ]


        if not selected_row.empty:

            current_keywords = str(

                selected_row.iloc[0][

                    "Keywords"

                ]

            )


            edited_vendor_name = st.text_input(

                "Vendor Name",

                value=selected_vendor,

                key="edit_vendor_name"

            )


            edited_keywords = st.text_input(

                "Keywords",

                value=current_keywords,

                key="edit_vendor_keywords"

            )


            if st.button(

                "✏️ Update Vendor",

                key="update_vendor"

            ):

                if not edited_vendor_name.strip():

                    st.warning(

                        "Vendor Name cannot be empty."

                    )

                elif not edited_keywords.strip():

                    st.warning(

                        "Keywords cannot be empty."

                    )

                else:

                    vendor_df = vendor_df[

                        vendor_df[

                            "Vendor Name"

                        ]

                        != selected_vendor

                    ]


                    new_row = pd.DataFrame({

                        "Vendor Name": [

                            edited_vendor_name.strip()

                        ],

                        "Keywords": [

                            edited_keywords.strip()

                        ]

                    })


                    vendor_df = pd.concat(

                        [

                            vendor_df,

                            new_row

                        ],

                        ignore_index=True

                    )


                    vendor_df.to_csv(

                        VENDOR_FILE,

                        index=False

                    )


                    st.success(

                        "✅ Vendor updated successfully!"

                    )


                    st.rerun()


    # ========================================================
    # DELETE VENDOR
    # ========================================================

    st.markdown("---")

    st.markdown(

        "### 🗑️ Delete Vendor"

    )


    if not vendor_df.empty:

        delete_vendor = st.selectbox(

            "Select Vendor to Delete",

            vendor_df[

                "Vendor Name"

            ].tolist(),

            key="delete_vendor_select"

        )


        confirm_delete = st.checkbox(

            "I confirm that I want to delete this vendor.",

            key="confirm_delete"

        )


        if st.button(

            "🗑️ Delete Vendor",

            key="delete_vendor"

        ):

            if not confirm_delete:

                st.warning(

                    "Please confirm deletion first."

                )

            else:

                vendor_df = vendor_df[

                    vendor_df[

                        "Vendor Name"

                    ]

                    != delete_vendor

                ]


                vendor_df.to_csv(

                    VENDOR_FILE,

                    index=False

                )


                st.success(

                    f"✅ {delete_vendor} deleted successfully!"

                )


                st.rerun()