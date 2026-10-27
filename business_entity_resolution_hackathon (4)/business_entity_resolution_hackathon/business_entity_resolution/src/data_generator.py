
from pathlib import Path
import pandas as pd
import random, re

random.seed(42)

BASE_BUSINESSES = [
    ("B001","ABC Technologies Private Limited","12 Gandhi Road, Chennai","India"),
    ("B002","Sri Lakshmi Hospital","45 MG Road, Coimbatore","India"),
    ("B003","Green Leaf Restaurant","18 Park Street, Kolkata","India"),
    ("B004","Tata Consultancy Services Limited","9 Mount Road, Chennai","India"),
    ("B005","Sunrise Electronics Corporation","72 Market Road, Mumbai","India"),
    ("B006","Blue Star Logistics Private Limited","21 Ring Road, Delhi","India"),
    ("B007","Global Textiles and Garments","8 Industrial Estate, Surat","India"),
    ("B008","River View Hotel","4 Lake Road, Kochi","India"),
    ("B009","Infosys Technologies Limited","44 IT Park Road, Bengaluru","India"),
    ("B010","Apollo Pharmacy Limited","16 Main Street, Hyderabad","India"),
    ("B011","Amazon Web Services Inc","410 Terry Avenue, Seattle","United States"),
    ("B012","Microsoft Corporation","One Microsoft Way, Redmond","United States"),
    ("B013","North Star Cafe","22 King Street, Pune","India"),
    ("B014","Metro Auto Parts Private Limited","5 Station Road, Madurai","India"),
    ("B015","Fresh Basket Supermarket","101 Central Avenue, Trichy","India"),
]

def clean(x):
    return re.sub(r"\s+", " ", x).strip()

def name_variant(name, i):
    reps = [
        ("Private Limited","Pvt Ltd"), ("Private Limited","Private Ltd"),
        ("Limited","Ltd"), ("Corporation","Corp"), ("Technologies","Tech"),
        ("and","&")
    ]
    out = name
    if i % 3 == 0:
        for a,b in reps:
            out = out.replace(a,b)
    if i % 4 == 0:
        out = out.replace(" ", "  ")
    if i % 5 == 0:
        out = out.upper()
    if i % 7 == 0:
        out = out.replace(".", "")
    return clean(out)

def address_variant(addr, i):
    out = addr
    for a,b in [("Road","Rd"),("Street","St"),("Avenue","Ave")]:
        if i % 2 == 0:
            out = out.replace(a,b)
    if i % 4 == 0:
        parts = out.split(",")
        out = ", ".join(reversed(parts))
    if i % 6 == 0:
        out = out.replace(", ", " ")
    return clean(out)

def generate_demo_dataset(base):
    train = base/"dataset/train"
    test = base/"dataset/test"
    train.mkdir(parents=True, exist_ok=True); test.mkdir(parents=True, exist_ok=True)

    s1, s2, s3, truth = [], [], [], []
    counters = {"S1":1,"S2":1,"S3":1}

    def sid(prefix):
        v = f"{prefix}-{counters[prefix]:05d}"; counters[prefix]+=1; return v

    # Training: each master gets 1-2 noisy links in S2/S3, plus distractors.
    for idx, (_, name, addr, country) in enumerate(BASE_BUSINESSES):
        a = sid("S1")
        s1.append((a,name,addr,country))
        matches=[]
        if idx % 5 != 0:
            b=sid("S2"); s2.append((b,name_variant(name,idx+1),address_variant(addr,idx+2),country)); matches.append(b)
        if idx % 3 != 0:
            c=sid("S3"); s3.append((c,name_variant(name,idx+3),address_variant(addr,idx+4),country)); matches.append(c)
        truth.append((a,",".join(matches)))

    # Add distractors with similar-looking names/addresses.
    distractors = [
        ("ABC Technologies","12 Gandhi Road, Chennai","India"),
        ("Green Leaf Cafe","18 Park Street, Kolkata","India"),
        ("Sunrise Electric Corp","72 Market Road, Mumbai","India"),
        ("Metro Auto Service","5 Station Road, Madurai","India"),
        ("Amazon Web Services","Seattle Downtown","United States"),
        ("North Star Coffee","23 King Street, Pune","India"),
    ]
    for i,(n,a,c) in enumerate(distractors):
        s2.append((sid("S2"), n, a, c))
        s3.append((sid("S3"), n, a, c))

    pd.DataFrame(s1,columns=["entity_id","business_name","business_address","country"]).to_csv(train/"train_source1.tsv",sep="\t",index=False)
    pd.DataFrame(s2,columns=["entity_id","business_name","business_address","country"]).to_csv(train/"train_source2.tsv",sep="\t",index=False)
    pd.DataFrame(s3,columns=["entity_id","business_name","business_address","country"]).to_csv(train/"train_source3.tsv",sep="\t",index=False)
    pd.DataFrame(truth,columns=["source1_entity_id","matched_entity_ids"]).to_csv(train/"train_ground_truth.tsv",sep="\t",index=False)

    # Test set: fresh IDs, noisy matches, and distractors.
    test_s1=[]; test_s2=[]; test_s3=[]
    for idx, (_, name, addr, country) in enumerate(BASE_BUSINESSES):
        a=sid("S1")
        test_s1.append((a,name,addr,country))
        if idx % 4 != 0:
            b=sid("S2"); test_s2.append((b,name_variant(name,idx+10),address_variant(addr,idx+11),country))
        if idx % 2 == 0:
            c=sid("S3"); test_s3.append((c,name_variant(name,idx+12),address_variant(addr,idx+13),country))
    for i,(n,a,c) in enumerate(distractors):
        test_s2.append((sid("S2"), n, a, c))
        test_s3.append((sid("S3"), n, a, c))

    pd.DataFrame(test_s1,columns=["entity_id","business_name","business_address","country"]).to_csv(test/"test_source1.tsv",sep="\t",index=False)
    pd.DataFrame(test_s2,columns=["entity_id","business_name","business_address","country"]).to_csv(test/"test_source2.tsv",sep="\t",index=False)
    pd.DataFrame(test_s3,columns=["entity_id","business_name","business_address","country"]).to_csv(test/"test_source3.tsv",sep="\t",index=False)

    return {
        "message":"Demo dataset generated",
        "train_source1":len(s1),"train_source2":len(s2),"train_source3":len(s3),
        "test_source1":len(test_s1),"test_source2":len(test_s2),"test_source3":len(test_s3),
        "ground_truth":len(truth)
    }
