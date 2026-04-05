"""
generator.py — Realistic Indian UPI transaction datasets.

Creates three sets of transactions:
  - Easy   (20 transactions): Clear merchant names, no fraud
  - Medium (40 transactions): Mixed names, some duplicates, some fraud
  - Hard   (60 transactions): Cryptic IDs, multiple fraud patterns, complex P2P

All transactions are seeded for reproducibility.
"""

from env.models import UPITransaction, TransactionType, Category, FlagType
from typing import List


# ─────────────────────────────────────────────────────────────
# EASY TASK — 20 transactions, obvious merchants, no fraud
# Agent just needs to categorize correctly
# ─────────────────────────────────────────────────────────────

def get_easy_transactions() -> List[UPITransaction]:
    return [
        UPITransaction(
            id="TXN001", amount=349.0, merchant_name="Swiggy",
            upi_id="swiggy@icici", timestamp="2024-01-15T12:30:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00001",
            ground_truth_category=Category.FOOD_DELIVERY,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN002", amount=199.0, merchant_name="Netflix India",
            upi_id="netflix@ybl", timestamp="2024-01-15T09:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Monthly subscription", bank_ref="HDFC00002",
            ground_truth_category=Category.SUBSCRIPTION,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        UPITransaction(
            id="TXN003", amount=250.0, merchant_name="Ola Cabs",
            upi_id="ola@axisbank", timestamp="2024-01-15T08:15:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00003",
            ground_truth_category=Category.TRANSPORT,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN004", amount=1200.0, merchant_name="BigBasket",
            upi_id="bigbasket@okaxis", timestamp="2024-01-14T18:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Weekly groceries", bank_ref="HDFC00004",
            ground_truth_category=Category.GROCERIES,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN005", amount=85000.0, merchant_name="SALARY CREDIT",
            upi_id="employer@hdfcbank", timestamp="2024-01-01T09:00:00",
            transaction_type=TransactionType.CREDIT,
            description="January salary", bank_ref="HDFC00005",
            ground_truth_category=Category.SALARY,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN006", amount=499.0, merchant_name="Spotify",
            upi_id="spotify@icici", timestamp="2024-01-05T10:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00006",
            ground_truth_category=Category.ENTERTAINMENT,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        UPITransaction(
            id="TXN007", amount=2400.0, merchant_name="BESCOM",
            upi_id="bescom@paytm", timestamp="2024-01-10T11:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Electricity bill", bank_ref="HDFC00007",
            ground_truth_category=Category.UTILITIES,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN008", amount=450.0, merchant_name="Zomato",
            upi_id="zomato@kotak", timestamp="2024-01-16T20:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00008",
            ground_truth_category=Category.FOOD_DELIVERY,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN009", amount=3500.0, merchant_name="Myntra",
            upi_id="myntra@ybl", timestamp="2024-01-13T15:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00009",
            ground_truth_category=Category.SHOPPING,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN010", amount=500.0, merchant_name="Apollo Pharmacy",
            upi_id="apollopharmacy@hdfcbank", timestamp="2024-01-12T14:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Medicines", bank_ref="HDFC00010",
            ground_truth_category=Category.MEDICAL,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN011", amount=320.0, merchant_name="Swiggy",
            upi_id="swiggy@icici", timestamp="2024-01-17T13:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00011",
            ground_truth_category=Category.FOOD_DELIVERY,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN012", amount=800.0, merchant_name="IRCTC",
            upi_id="irctc@sbi", timestamp="2024-01-11T16:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Train ticket", bank_ref="HDFC00012",
            ground_truth_category=Category.TRANSPORT,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN013", amount=999.0, merchant_name="Amazon Prime",
            upi_id="amazon@apl", timestamp="2024-01-08T10:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Annual subscription", bank_ref="HDFC00013",
            ground_truth_category=Category.SUBSCRIPTION,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        UPITransaction(
            id="TXN014", amount=349.0, merchant_name="Swiggy",
            upi_id="swiggy@icici", timestamp="2024-01-18T19:30:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00014",
            ground_truth_category=Category.FOOD_DELIVERY,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN015", amount=15000.0, merchant_name="Priya Sharma",
            upi_id="priyasharma@okicici", timestamp="2024-01-01T10:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Rent January", bank_ref="HDFC00015",
            ground_truth_category=Category.RENT,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        UPITransaction(
            id="TXN016", amount=2999.0, merchant_name="Coursera",
            upi_id="coursera@paypal", timestamp="2024-01-09T12:00:00",
            transaction_type=TransactionType.DEBIT,
            description="ML course", bank_ref="HDFC00016",
            ground_truth_category=Category.EDUCATION,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN017", amount=5000.0, merchant_name="Zerodha",
            upi_id="zerodha@kotak", timestamp="2024-01-10T09:30:00",
            transaction_type=TransactionType.DEBIT,
            description="SIP investment", bank_ref="HDFC00017",
            ground_truth_category=Category.INVESTMENT,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN018", amount=349.0, merchant_name="Swiggy",
            upi_id="swiggy@icici", timestamp="2024-01-15T12:31:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00018",
            ground_truth_category=Category.FOOD_DELIVERY,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN019", amount=180.0, merchant_name="Uber",
            upi_id="uber@axisbank", timestamp="2024-01-16T22:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00019",
            ground_truth_category=Category.TRANSPORT,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN020", amount=349.0, merchant_name="Swiggy REFUND",
            upi_id="swiggy@icici", timestamp="2024-01-19T10:00:00",
            transaction_type=TransactionType.CREDIT,
            description="Order cancelled refund", bank_ref="HDFC00020",
            ground_truth_category=Category.REFUND,
            ground_truth_flag=FlagType.NORMAL
        ),
    ]


# ─────────────────────────────────────────────────────────────
# MEDIUM TASK — 40 transactions
# Mix of clear and ambiguous, duplicates, some fraud
# ─────────────────────────────────────────────────────────────

def get_medium_transactions() -> List[UPITransaction]:
    easy = get_easy_transactions()

    extra = [
        # Ambiguous P2P — could be rent or transfer
        UPITransaction(
            id="TXN021", amount=15000.0, merchant_name="Amit Kumar",
            upi_id="9876543210@paytm", timestamp="2024-02-01T10:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00021",
            ground_truth_category=Category.RENT,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        # Duplicate of TXN001
        UPITransaction(
            id="TXN022", amount=349.0, merchant_name="Swiggy",
            upi_id="swiggy@icici", timestamp="2024-01-15T12:32:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00022",
            ground_truth_category=Category.FOOD_DELIVERY,
            ground_truth_flag=FlagType.DUPLICATE
        ),
        # Suspicious — odd amount at 3am
        UPITransaction(
            id="TXN023", amount=9999.0, merchant_name="UNKNOWN MERCHANT",
            upi_id="unknown123@ybl", timestamp="2024-01-20T03:14:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00023",
            ground_truth_category=Category.SUSPICIOUS,
            ground_truth_flag=FlagType.SUSPICIOUS
        ),
        # Bill split with friend
        UPITransaction(
            id="TXN024", amount=750.0, merchant_name="Rohit Verma",
            upi_id="rohitv@okicici", timestamp="2024-01-21T21:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Dinner split", bank_ref="HDFC00024",
            ground_truth_category=Category.BILL_SPLIT,
            ground_truth_flag=FlagType.NORMAL
        ),
        # Forgotten subscription
        UPITransaction(
            id="TXN025", amount=129.0, merchant_name="Hotstar",
            upi_id="hotstar@axisbank", timestamp="2024-01-05T09:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00025",
            ground_truth_category=Category.SUBSCRIPTION,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        UPITransaction(
            id="TXN026", amount=250.0, merchant_name="Meera Nair",
            upi_id="meeran@okhdfcbank", timestamp="2024-01-22T18:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Borrowed money", bank_ref="HDFC00026",
            ground_truth_category=Category.TRANSFER,
            ground_truth_flag=FlagType.NORMAL
        ),
        # Another suspicious transaction
        UPITransaction(
            id="TXN027", amount=49999.0, merchant_name="PRIZE WINNER",
            upi_id="prize@paytm", timestamp="2024-01-23T15:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Prize claim fee", bank_ref="HDFC00027",
            ground_truth_category=Category.SUSPICIOUS,
            ground_truth_flag=FlagType.SUSPICIOUS
        ),
        UPITransaction(
            id="TXN028", amount=600.0, merchant_name="DMart",
            upi_id="dmart@hdfcbank", timestamp="2024-01-20T17:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00028",
            ground_truth_category=Category.GROCERIES,
            ground_truth_flag=FlagType.NORMAL
        ),
        # Duplicate of TXN007
        UPITransaction(
            id="TXN029", amount=2400.0, merchant_name="BESCOM",
            upi_id="bescom@paytm", timestamp="2024-01-10T11:05:00",
            transaction_type=TransactionType.DEBIT,
            description="Electricity bill", bank_ref="HDFC00029",
            ground_truth_category=Category.UTILITIES,
            ground_truth_flag=FlagType.DUPLICATE
        ),
        UPITransaction(
            id="TXN030", amount=199.0, merchant_name="Sunita Devi",
            upi_id="sunitad@okaxis", timestamp="2024-01-24T12:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Vegetables", bank_ref="HDFC00030",
            ground_truth_category=Category.GROCERIES,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN031", amount=1500.0, merchant_name="Flipkart",
            upi_id="flipkart@ybl", timestamp="2024-01-25T14:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00031",
            ground_truth_category=Category.SHOPPING,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN032", amount=85000.0, merchant_name="SALARY CREDIT",
            upi_id="employer@hdfcbank", timestamp="2024-02-01T09:00:00",
            transaction_type=TransactionType.CREDIT,
            description="February salary", bank_ref="HDFC00032",
            ground_truth_category=Category.SALARY,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN033", amount=300.0, merchant_name="Rahul Singh",
            upi_id="rahuls@paytm", timestamp="2024-01-26T20:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Movie tickets split", bank_ref="HDFC00033",
            ground_truth_category=Category.BILL_SPLIT,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN034", amount=450.0, merchant_name="Zomato",
            upi_id="zomato@kotak", timestamp="2024-01-27T19:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00034",
            ground_truth_category=Category.FOOD_DELIVERY,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN035", amount=1200.0, merchant_name="Max Healthcare",
            upi_id="maxhealthcare@hdfcbank", timestamp="2024-01-28T11:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Doctor consultation", bank_ref="HDFC00035",
            ground_truth_category=Category.MEDICAL,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN036", amount=99.0, merchant_name="Google One",
            upi_id="google@okicici", timestamp="2024-01-05T08:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Storage subscription", bank_ref="HDFC00036",
            ground_truth_category=Category.SUBSCRIPTION,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        UPITransaction(
            id="TXN037", amount=3000.0, merchant_name="Ananya Krishnan",
            upi_id="ananyak@okaxis", timestamp="2024-01-29T10:00:00",
            transaction_type=TransactionType.CREDIT,
            description="Returned money", bank_ref="HDFC00037",
            ground_truth_category=Category.TRANSFER,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN038", amount=180.0, merchant_name="Uber",
            upi_id="uber@axisbank", timestamp="2024-01-30T08:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00038",
            ground_truth_category=Category.TRANSPORT,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN039", amount=10000.0, merchant_name="Zerodha",
            upi_id="zerodha@kotak", timestamp="2024-01-31T09:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Lumpsum investment", bank_ref="HDFC00039",
            ground_truth_category=Category.INVESTMENT,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN040", amount=500.0, merchant_name="Vikram Nair",
            upi_id="vikramn@paytm", timestamp="2024-01-31T22:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00040",
            ground_truth_category=Category.TRANSFER,
            ground_truth_flag=FlagType.NORMAL
        ),
    ]

    return easy + extra


# ─────────────────────────────────────────────────────────────
# HARD TASK — 60 transactions
# Cryptic names, complex fraud, hidden patterns
# ─────────────────────────────────────────────────────────────

def get_hard_transactions() -> List[UPITransaction]:
    medium = get_medium_transactions()

    extra = [
        # Cryptic POS transactions
        UPITransaction(
            id="TXN041", amount=342.0, merchant_name="POS/TXN/884432",
            upi_id="pos884432@hdfcbank", timestamp="2024-02-02T13:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00041",
            ground_truth_category=Category.SHOPPING,
            ground_truth_flag=FlagType.NORMAL
        ),
        # Phishing — looks like IRCTC refund but is debit
        UPITransaction(
            id="TXN042", amount=1200.0, merchant_name="IRCTC-REFUND",
            upi_id="irctcrefund@paytm", timestamp="2024-02-03T02:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Refund processing fee", bank_ref="HDFC00042",
            ground_truth_category=Category.SUSPICIOUS,
            ground_truth_flag=FlagType.SUSPICIOUS
        ),
        # Phone number only
        UPITransaction(
            id="TXN043", amount=15000.0, merchant_name="8765432109",
            upi_id="8765432109@okicici", timestamp="2024-03-01T10:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00043",
            ground_truth_category=Category.RENT,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        # Recurring subscription — easily missed
        UPITransaction(
            id="TXN044", amount=49.0, merchant_name="AUTOPAY/LINKEDIN",
            upi_id="linkedin@axisbank", timestamp="2024-02-05T00:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00044",
            ground_truth_category=Category.SUBSCRIPTION,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        # Fraud — small test charge before big fraud
        UPITransaction(
            id="TXN045", amount=1.0, merchant_name="VERIFY/CARD",
            upi_id="verify@ybl", timestamp="2024-02-06T04:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00045",
            ground_truth_category=Category.SUSPICIOUS,
            ground_truth_flag=FlagType.SUSPICIOUS
        ),
        # Large fraud following the test charge
        UPITransaction(
            id="TXN046", amount=24999.0, merchant_name="ONLINE STORE XYZ",
            upi_id="storexyz@paytm", timestamp="2024-02-06T04:05:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00046",
            ground_truth_category=Category.SUSPICIOUS,
            ground_truth_flag=FlagType.SUSPICIOUS
        ),
        # Cryptic — actually a fuel station
        UPITransaction(
            id="TXN047", amount=2500.0, merchant_name="HP/PETROL/STATION",
            upi_id="hppetrol@okaxis", timestamp="2024-02-07T08:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00047",
            ground_truth_category=Category.TRANSPORT,
            ground_truth_flag=FlagType.NORMAL
        ),
        # Duplicate with slightly different amount (common fraud)
        UPITransaction(
            id="TXN048", amount=15001.0, merchant_name="8765432109",
            upi_id="8765432109@okicici", timestamp="2024-03-01T10:02:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00048",
            ground_truth_category=Category.SUSPICIOUS,
            ground_truth_flag=FlagType.SUSPICIOUS
        ),
        # Money from family — ambiguous
        UPITransaction(
            id="TXN049", amount=20000.0, merchant_name="Suresh Kumar",
            upi_id="sureshk@sbi", timestamp="2024-02-08T11:00:00",
            transaction_type=TransactionType.CREDIT,
            description="From dad", bank_ref="HDFC00049",
            ground_truth_category=Category.TRANSFER,
            ground_truth_flag=FlagType.NORMAL
        ),
        # Cryptic but actually grocery
        UPITransaction(
            id="TXN050", amount=875.0, merchant_name="QR/KIRANA/STORE",
            upi_id="kirana123@paytm", timestamp="2024-02-09T10:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00050",
            ground_truth_category=Category.GROCERIES,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN051", amount=85000.0, merchant_name="SALARY CREDIT",
            upi_id="employer@hdfcbank", timestamp="2024-03-01T09:00:00",
            transaction_type=TransactionType.CREDIT,
            description="March salary", bank_ref="HDFC00051",
            ground_truth_category=Category.SALARY,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN052", amount=199.0, merchant_name="NF*NETFLIX",
            upi_id="netflix@ybl", timestamp="2024-03-05T09:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00052",
            ground_truth_category=Category.SUBSCRIPTION,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        # Suspicious — KYC update scam
        UPITransaction(
            id="TXN053", amount=5.0, merchant_name="HDFC KYC UPDATE",
            upi_id="hdfckyc@paytm", timestamp="2024-02-10T15:00:00",
            transaction_type=TransactionType.DEBIT,
            description="KYC verification fee", bank_ref="HDFC00053",
            ground_truth_category=Category.SUSPICIOUS,
            ground_truth_flag=FlagType.SUSPICIOUS
        ),
        UPITransaction(
            id="TXN054", amount=450.0, merchant_name="AUTO/PAY/OLA",
            upi_id="ola@axisbank", timestamp="2024-02-11T09:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00054",
            ground_truth_category=Category.TRANSPORT,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN055", amount=2000.0, merchant_name="Neha Gupta",
            upi_id="nehag@okicici", timestamp="2024-02-14T20:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Valentine dinner split", bank_ref="HDFC00055",
            ground_truth_category=Category.BILL_SPLIT,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN056", amount=749.0, merchant_name="SWGY*ORDER",
            upi_id="swiggy@icici", timestamp="2024-02-15T20:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00056",
            ground_truth_category=Category.FOOD_DELIVERY,
            ground_truth_flag=FlagType.NORMAL
        ),
        # Duplicate of TXN056 within seconds
        UPITransaction(
            id="TXN057", amount=749.0, merchant_name="SWGY*ORDER",
            upi_id="swiggy@icici", timestamp="2024-02-15T20:00:45",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00057",
            ground_truth_category=Category.FOOD_DELIVERY,
            ground_truth_flag=FlagType.DUPLICATE
        ),
        UPITransaction(
            id="TXN058", amount=3500.0, merchant_name="AMZN*MKTP",
            upi_id="amazon@apl", timestamp="2024-02-16T14:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00058",
            ground_truth_category=Category.SHOPPING,
            ground_truth_flag=FlagType.NORMAL
        ),
        UPITransaction(
            id="TXN059", amount=15000.0, merchant_name="Priya Sharma",
            upi_id="priyasharma@okicici", timestamp="2024-02-01T10:00:00",
            transaction_type=TransactionType.DEBIT,
            description=None, bank_ref="HDFC00059",
            ground_truth_category=Category.RENT,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
        UPITransaction(
            id="TXN060", amount=299.0, merchant_name="AUTOPAY/CURE.FIT",
            upi_id="curefit@axisbank", timestamp="2024-02-01T07:00:00",
            transaction_type=TransactionType.DEBIT,
            description="Gym subscription", bank_ref="HDFC00060",
            ground_truth_category=Category.SUBSCRIPTION,
            ground_truth_flag=FlagType.NORMAL,
            is_recurring=True
        ),
    ]

    return medium + extra


# ─────────────────────────────────────────────────────────────
# TASK LOADER
# ─────────────────────────────────────────────────────────────

def load_transactions(difficulty: str) -> List[UPITransaction]:
    if difficulty == "easy":
        return get_easy_transactions()
    elif difficulty == "medium":
        return get_medium_transactions()
    elif difficulty == "hard":
        return get_hard_transactions()
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
