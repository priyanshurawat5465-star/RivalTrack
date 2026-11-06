import os
import pandas as pd

RAW_DATA_PATH = "data/raw_scraped_corpus.csv"
CLEAN_DATA_PATH = "data/cleaned_absa_corpus.csv"

# Add any Electronics URLs here that were scraped while tagged as "Footwear"
MISLABELED_ELECTRONICS_URLS = [
    "https://www.flipkart.com/apple-iphone-15-black-128-gb/product-reviews/itm3234701b0863a?pid=MOBHQV9YR4CBTD9D&lid=LSTMOBHQV9YR4CBTD9DPLROKI&marketplace=FLIPKART&pageUID=1791128918014",
"https://www.flipkart.com/samsung-galaxy-s24-5g-snapdragon-onyx-black-128-gb/product-reviews/itm3469a7107606f?pid=MOBHDVFKSSHPUYHB&lid=LSTMOBHDVFKSSHPUYHBXUF3VR&marketplace=FLIPKART&pageUID=1791129185286",
"https://www.flipkart.com/samsung-galaxy-f14-5g-b-a-e-purple-128-gb/product-reviews/itmdc1c60aa887ff?pid=MOBGNBFNNZER2WZW&lid=LSTMOBGNBFNNZER2WZWDG8CJ0&marketplace=FLIPKART&pageUID=1791129359446",
"https://www.flipkart.com/samsung-galaxy-f05-twilight-blue-64-gb/product-reviews/itm84a914081ab93?pid=MOBH4GTXZ9ZRQ3SU&lid=LSTMOBH4GTXZ9ZRQ3SUXCBGGE&marketplace=FLIPKART&pageUID=1791129280096",
"https://www.flipkart.com/redmi-note-9-arctic-white-64-gb/product-reviews/itm4ab07b858925c?pid=MOBFU3ZFXMWJCSHP&lid=LSTMOBFU3ZFXMWJCSHPXQ6FTB&marketplace=FLIPKART&pageUID=1791129772207",
"https://www.flipkart.com/redmi-8-ruby-red-64-gb/product-reviews/itmf96d708d88b1b?pid=MOBFKPYDCVSCZBYR&lid=LSTMOBFKPYDCVSCZBYRSUFG5C&marketplace=FLIPKART&pageUID=1791129823706",
"https://www.flipkart.com/redmi-10-pacific-blue-64-gb/product-reviews/itm0f2a6a2112b75?pid=MOBGC9GYCHQZK9GW&lid=LSTMOBGC9GYCHQZK9GWLVQCAU&marketplace=FLIPKART&pageUID=1791129869974",
"https://www.flipkart.com/motorola-g35-5g-midnight-black-128-gb/product-reviews/itm015cbd184b0eb?pid=MOBH3YGPTSA4SZQF&lid=LSTMOBH3YGPTSA4SZQFL1PUJJ&marketplace=FLIPKART&pageUID=1791129931437",
"https://www.flipkart.com/mivi-play-12hrs-playback-bass-boosted-tws-feature-ipx4-5-w-portable-bluetooth-speaker/product-reviews/itm9dcdfcd2431db?pid=ACCG25T3Z4CHBUAP&lid=LSTACCG25T3Z4CHBUAPQJ9CJC&marketplace=FLIPKART&pageUID=1791130082642",
"https://www.flipkart.com/mivi-fort-s60-2-inbuilt-subwoofers-3eq-modes-remote-ctrl-60-w-bluetooth-soundbar/product-reviews/itm25169429bfb89?pid=ACCGAU6GRGWZCCVQ&lid=LSTACCGAU6GRGWZCCVQXWNBBO&marketplace=FLIPKART&pageUID=1791130144163",
"https://www.flipkart.com/boat-wave-fury-smartwatch/product-reviews/itm8a785ea5801dc?pid=SMWGQHN4MWU6ZJGN&lid=LSTSMWGQHN4MWU6ZJGNRJO52S&marketplace=FLIPKART&pageUID=1791130472935",
"https://www.flipkart.com/boat-stone-650-10-w-bluetooth-speaker/product-reviews/itm3d7a934f4b632?pid=ACCFDBH8QNDC4BMZ&lid=LSTACCFDBH8QNDC4BMZV2RAJG&marketplace=FLIPKART&pageUID=1791130250482",
"https://www.flipkart.com/portronics-sounddrum-1-12w-tws-portable-type-c-charging-12-w-bluetooth-speaker/product-reviews/itmec32d68debb98?pid=ACCG3Y5YNHVGZDRE&lid=LSTACCG3Y5YNHVGZDREUSUDOF&marketplace=FLIPKART&pageUID=1791130284910",
"https://www.flipkart.com/boat-storm-call-3-w-tbt-navigation-watch-face-studio-1-83-4-6-cm-hd-display-qr-tray-smartwatch/product-reviews/itm586335e08aa88?pid=SMWGXWZDAAFJXCXB&lid=LSTSMWGXWZDAAFJXCXBJAG2M1&marketplace=FLIPKART&pageUID=1791130370905",
"https://www.flipkart.com/goboult-drift-bt-calling-hd-display-140-watchface-calling-ip68-smartwatch/product-reviews/itm50c9247c6b13c?pid=SMWGF5GETH3AH6WG&lid=LSTSMWGF5GETH3AH6WGVIN7BI&marketplace=FLIPKART&pageUID=1791130577191",
"https://www.flipkart.com/realme-watch-5-1-97-amoled-upto-20-days-battery-bt-calling-gps-compass-ip68-smartwatch/product-reviews/itmf3a780e7cb90d?pid=SMWHHKZXBJWEP8G7&lid=LSTSMWHHKZXBJWEP8G7XTOW22&marketplace=FLIPKART&pageUID=1791130668685",
"https://www.flipkart.com/mi-10000-mah-22-5-w-slim-pocket-size-power-bank/product-reviews/itm0bbd6e648368b?pid=PWBH2GGFDGURFAK9&lid=LSTPWBH2GGFDGURFAK9ANPROK&marketplace=FLIPKART&pageUID=1791130941330",
"https://www.flipkart.com/apple-iphone-12-white-64-gb/product-reviews/itm8b88bdc03cd79?pid=MOBFWBYZTK33MBG9&lid=LSTMOBFWBYZTK33MBG9YC4F9Y&marketplace=FLIPKART&pageUID=1791131110855",
"https://www.flipkart.com/apple-iphone-16-black-128-gb/product-reviews/itmb07d67f995271?pid=MOBH4DQFG8NKFRDY&lid=LSTMOBH4DQFG8NKFRDYNBDOZI&marketplace=FLIPKART&pageUID=1791131144774",
"https://www.flipkart.com/mivi-duopods-k2-40hrs-pt-hd-call-fast-charging-50ms-latency-ipx4-voice-asst-5-3v-bluetooth/product-reviews/itm6e514d29fd947?pid=ACCGPFCN6ZKBF5MD&lid=LSTACCGPFCN6ZKBF5MDRNL8SQ&marketplace=FLIPKART&pageUID=1791131193338",
"https://www.flipkart.com/mivi-duopods-i6-55hrs-pt-hd-call-fast-charging-3d-sound-9eq-modes-app-supp-ipx4-bluetooth/product-reviews/itm0bd6123928091?pid=ACCH238YUFZPUXZM&lid=LSTACCH238YUFZPUXZMLLBOGT&marketplace=FLIPKART&pageUID=1791131407572",
"https://www.flipkart.com/triggr-kraken-x4-13mm-drivers-40ms-latency-enc-60h-battery-rubber-finish-v5-4-bluetooth-gaming/product-reviews/itm11fd34f2ddfb6?pid=ACCH2WWTZUWTRNHP&lid=LSTACCH2WWTZUWTRNHPZ7SX0C&marketplace=FLIPKART&pageUID=1791131433316",
"https://www.flipkart.com/aroma-nb121-pods-upto-40-hours-playtime-type-c-fast-charging-dual-pairing-earbuds-bluetooth/product-reviews/itm911b47c91c5dd?pid=ACCH77MF4NYVHZTE&lid=LSTACCH77MF4NYVHZTETGQDWP&marketplace=FLIPKART&pageUID=1791131461910",
"https://www.flipkart.com/oneplus-nord-buds-3r-tws-earbud-10min-charge-8hr-music-3d-audio-2-mic-call-bluetooth/product-reviews/itm87ffbf2a0bf08?pid=ACCHF3U4VUQCEGQ5&lid=LSTACCHF3U4VUQCEGQ5H3MOIP&marketplace=FLIPKART&pageUID=1791131492255",
"https://www.flipkart.com/infinix-note-30-5g-magic-black-256-gb/product-reviews/itm12c179f0a6281?pid=MOBGQ9HPEZKFHZFQ&lid=LSTMOBGQ9HPEZKFHZFQUVHIKF&marketplace=FLIPKART&pageUID=1791131582117",
"https://www.flipkart.com/nokia-105-single-sim-keypad-mobile-phone-wireless-fm-radio/product-reviews/itm4a32eadacb09b?pid=MOBGC96FVTHVNXNH&lid=LSTMOBGC96FVTHVNXNHCSXH4E&marketplace=FLIPKART&pageUID=1791131735076",
"https://www.flipkart.com/vivo-t4r-5g-twilight-blue-256-gb/product-reviews/itm3a5653667cb9c?pid=MOBHE49GFMXQ5QEG&lid=LSTMOBHE49GFMXQ5QEGZPB4G9&marketplace=FLIPKART&pageUID=1791131806468",
"https://www.flipkart.com/boat-20000-mah-22-5-w-compact-pocket-size-power-bank/product-reviews/itm812d8194c424f?pid=PWBGR4KGUMSZQMFZ&lid&marketplace=FLIPKART",
"https://www.flipkart.com/ambrane-20000-mah-22-5-w-compact-pocket-size-power-bank/product-reviews/itm1099e5d667635?pid=PWBHAYMGCNB8YTVK&lid&marketplace=FLIPKART",
"https://www.flipkart.com/urbn-10000-mah-22-5-w-compact-pocket-size-power-bank/product-reviews/itmf2afa6a01967d?pid=PWBHBCHYXH7ZTXHH&lid&marketplace=FLIPKART"
]

def clean_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found.")
        return

    print("--- Phase 2: Starting Data Preprocessing & Cleaning ---")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Initial raw row count: {len(df)}")

    # 1. Fix Mislabeled Categories
    if MISLABELED_ELECTRONICS_URLS:
        df.loc[df['product_url'].isin(MISLABELED_ELECTRONICS_URLS), 'category'] = 'Electronics'
        print("Updated mislabeled Electronics categories.")

    # 2. Text Normalization & Clean Artifacts
    df['review_text'] = df['review_text'].astype(str)
    df['review_text'] = df['review_text'].str.replace('READ MORE', '', case=False)
    df['review_text'] = df['review_text'].str.strip()
    df['review_text'] = df['review_text'].str.replace(r'\s+', ' ', regex=True)

    # 3. Filter Out Short or Empty Reviews
    df = df[df['review_text'].str.len() >= 20]

    # 4. Global Deduplication across Reviews
    df = df.drop_duplicates(subset=['review_text']).reset_index(drop=True)
    print(f"Rows after removing duplicate text: {len(df)}")

    # 5. Cap Reviews per Product (Max 250 per product listing)
    df = df.groupby('product_url').head(250).reset_index(drop=True)
    print(f"Rows after capping each product to max 250 reviews: {len(df)}")

    # 6. Re-verify Sentiment Binary Assignment (0: Rating <= 3.0, 1: Rating > 3.0)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    df['sentiment'] = (df['rating'] > 3.0).astype(int)

    # Save Cleaned Corpus
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\n Cleaned corpus successfully saved to: {CLEAN_DATA_PATH}")

    # --- Dataset Summary Statistics ---
    print("\n================ DATASET SUMMARY ================")
    print(f"Total Cleaned Rows: {len(df)}")
    print("\nRows per Category:")
    print(df['category'].value_counts())
    print("\nOverall Sentiment Breakdown:")
    print(df['sentiment'].value_counts(normalize=True).map(lambda n: f"{n:.1%}"))
    print("==================================================")

if __name__ == "__main__":
    clean_dataset()