import asyncio
import hashlib
import os
import random
import pandas as pd
from playwright.async_api import async_playwright, Route

# ---------------- CONFIGURATION ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "raw_scraped_corpus.csv")
MAX_REVIEWS_PER_PRODUCT = 250

os.makedirs(DATA_DIR, exist_ok=True)

CURRENT_CATEGORY = "Apparel"
SEED_URLS = [
"https://www.flipkart.com/vebnor-men-solid-casual-grey-shirt/product-reviews/itmfbb7c9d433df3?pid=SHTFZFTGUBEFPC7X&lid&marketplace=FLIPKART",
"https://www.flipkart.com/blue-squad-men-solid-casual-red-shirt/product-reviews/itme9698eb92d95b?pid=SHTH75HE7X76NZ7P&lid&marketplace=FLIPKART",
"https://www.flipkart.com/highlander-men-solid-casual-white-shirt/product-reviews/itm9336e1a90e312?pid=SHTEH5HRS9SYYXTJ&lid=&aid=overall&certifiedBuyer=false&sortOrder=MOST_HELPFUL&pageUID=1791175600530",
"https://www.flipkart.com/allwin-paul-men-checkered-casual-brown-shirt/product-reviews/itma5a3ebca3aec8?pid=SHTGFU69ZTVHFFNC&lid&marketplace=FLIPKART",
"https://www.flipkart.com/allen-solly-solid-men-polo-neck-maroon-t-shirt/product-reviews/itm5648b3eb638d6?pid=TSHGTBWVNSBQHHYU&lid&marketplace=FLIPKART",
"https://www.flipkart.com/ftx-solid-men-round-neck-silver-blue-black-grey-t-shirt/product-reviews/itm39275afebb5b2?pid=TSHGEMGVQKQFYMPY&lid&marketplace=FLIPKART",
"https://www.flipkart.com/try-colorblock-men-polo-neck-dark-green-white-t-shirt/product-reviews/itmd58efb47ab46e?pid=TSHHJQUPCMMYREUN&lid&marketplace=FLIPKART",
"https://www.flipkart.com/puma-solid-men-polo-neck-white-t-shirt/product-reviews/itm24c847dea3d76?pid=TSHH4D2FE9C2BSHM&lid&marketplace=FLIPKART",
"https://www.flipkart.com/zaysh-regular-men-black-jeans/product-reviews/itm362c51134df0c?pid=JEAF8H62ZJFGMFVF&lid=LSTJEAF8H62ZJFGMFVFIOMLS6&marketplace=FLIPKART&pageUID=1791176965012",
"https://www.flipkart.com/lzard-slim-men-dark-blue-jeans/product-reviews/itmdd82030b26cd3?pid=JEAGVVZJRRYHAHHY&lid=LSTJEAGVVZJRRYHAHHY3THKWD&marketplace=FLIPKART&pageUID=1791177003914",
"https://www.flipkart.com/highlander-tapered-fit-men-black-jeans/product-reviews/itm9e52b6573afe3?pid=JEAFEBZYHMKJHXEM&lid=LSTJEAFEBZYHMKJHXEML1IXGE&marketplace=FLIPKART&pageUID=1791177040202",
"https://www.flipkart.com/kashian-flared-women-light-blue-jeans/product-reviews/itmc41a264888f3d?pid=JEAHG7X895G6EG4Z&lid=LSTJEAHG7X895G6EG4ZGAD1AL&marketplace=FLIPKART&pageUID=1791177461881",
"https://www.flipkart.com/kotty-wide-leg-women-black-jeans/product-reviews/itm92c0a90d42859?pid=JEAHNP27PABANVTG&lid=LSTJEAHNP27PABANVTG4ZXJ6X&marketplace=FLIPKART&pageUID=1791177727405",
"https://www.flipkart.com/zayla-loose-fit-women-blue-jeans/product-reviews/itmfe49eeb35c426?pid=JEAH7CDZB2FHCWGG&lid=LSTJEAH7CDZB2FHCWGGMFWTFD&marketplace=FLIPKART&pageUID=1791177847554",
"https://www.flipkart.com/guti-flared-women-blue-jeans/product-reviews/itm9d67a6a8368c2?pid=JEAH3ZTFHYG5GZ2G&lid=LSTJEAH3ZTFHYG5GZ2GZ938EN&marketplace=FLIPKART&pageUID=1791177881647",
"https://www.flipkart.com/kotty-flared-women-blue-jeans/product-reviews/itma558d62fde67e?pid=JEAHZEGGQR5YUVKY&lid=LSTJEAHZEGGQR5YUVKYL25FMY&marketplace=FLIPKART&pageUID=1791178016752",
"https://www.flipkart.com/kotty-regular-women-black-jeans/product-reviews/itm756b360fe28e7?pid=JEAHZEGG83DCABSA&lid=LSTJEAHZEGG83DCABSAWESNM9&marketplace=FLIPKART&pageUID=1791178525443",
"https://www.flipkart.com/kotty-regular-fit-women-black-trousers/product-reviews/itm126ac29fac269?pid=TROGHF9AFFAHR8GP&lid&marketplace=FLIPKART",
"https://www.flipkart.com/kotty-loose-fit-women-beige-trousers/product-reviews/itmfedf32fe7ef36?pid=TROHATH3RGFDDKSJ&lid=LSTTROHATH3RGFDDKSJ8BSQQO&marketplace=FLIPKART",
"https://www.flipkart.com/combraided-slim-fit-men-black-trousers/product-reviews/itm128c8de6ef2cc?pid=TROHKNVYFNWYKZH3&lid&marketplace=FLIPKART",
"https://www.flipkart.com/indiclub-loose-fit-men-black-beige-trousers/product-reviews/itmbfa8f6dceec3e?pid=TROHHVYZRATXG7T8&lid&marketplace=FLIPKART",
"https://www.flipkart.com/mokosh-women-kurta-pant-dupatta-set/product-reviews/itma58836dbc240b?pid=ETHHCZ65PEE3UCGQ&lid=LSTETHHCZ65PEE3UCGQ8WYTPO&marketplace=FLIPKART&pageUID=1791179810094",
"https://www.flipkart.com/fabunna-cotton-women-kurta-pant-dupatta-set/product-reviews/itmb326efc0dc9ca?pid=ETHGZWFBF7B5NRRB&lid=LSTETHGZWFBF7B5NRRBV3JPI7&marketplace=FLIPKART&pageUID=1791179845654",
"https://www.flipkart.com/mishnee-fashion-women-kurti-palazzo-dupatta-set/product-reviews/itm5389fe0eecbaf?pid=ETHHJAYZBZNKMKDG&lid=LSTETHHJAYZBZNKMKDGZO6XRO&marketplace=FLIPKART&pageUID=1791180000695",
"https://www.flipkart.com/koshin-men-kurta-pyjama-set/product-reviews/itm942ed8deb7b13?pid=ETHGMWFNV4QPMTUY&lid=LSTETHGMWFNV4QPMTUY3DPTAS&marketplace=FLIPKART&pageUID=1791180117674",
"https://www.flipkart.com/elepants-women-kurti-palazzo-set/product-reviews/itm0e33512a8aa80?pid=ETHHM8GD2RPDGC6R&lid=LSTETHHM8GD2RPDGC6RRWN0QL&marketplace=FLIPKART&pageUID=1791180236474",
"https://www.flipkart.com/shree-ramkrishna-fab-women-kurti-pant-dupatta-set/product-reviews/itm92eae7b665ba2?pid=ETHHBPTNXKBDM4HH&lid=LSTETHHBPTNXKBDM4HH7JVK86&marketplace=FLIPKART&pageUID=1791180304820",
"https://www.flipkart.com/hittu-women-kurta-dupatta-set/product-reviews/itm4ac761cfd918e?pid=ETHGEJGSAVZ7AYZM&lid=LSTETHGEJGSAVZ7AYZMI9YWYP&marketplace=FLIPKART&pageUID=1791180408920",
"https://www.flipkart.com/tibra-collection-men-kurta-pyjama-set/product-reviews/itmcbbc5715b99bb?pid=ETHFZ6F6ZHF2ZKDG&lid=LSTETHFZ6F6ZHF2ZKDGFJK8M5&marketplace=FLIPKART&pageUID=1791180493781",
"https://www.flipkart.com/tibra-attire-women-kurta-pant-dupatta-set/product-reviews/itm1292c44c16e77?pid=ETHGW5G5ZWZPTKXZ&lid=LSTETHGW5G5ZWZPTKXZLFJO45&marketplace=FLIPKART&pageUID=1791180693113"
]

seen_hashes = set()

async def block_resources(route: Route):
    """Blocks images, media, and fonts to conserve cellular hotspot data."""
    if route.request.resource_type in ["image", "media", "font"]:
        await route.abort()
    else:
        await route.continue_()

def trigger_ip_rotation():
    print("\a\a\a")  
    print("\n[!] Block/Throttle Detected by Flipkart!")
    print("[!] 1. Turn ON Airplane Mode on your phone.")
    print("[!] 2. Wait 5-10 seconds.")
    print("[!] 3. Turn OFF Airplane Mode to get a new IP.")
    input("Press ENTER in this terminal when connection is restored...")

def get_text_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

async def scrape_flipkart():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False, 
            args=['--disable-dev-shm-usage', '--no-sandbox']
        )
        
        if not os.path.exists(DATA_FILE):
            pd.DataFrame(columns=["category", "product_url", "rating", "sentiment", "review_text"]).to_csv(DATA_FILE, index=False)

        total_rows = 0

        for product_idx, url in enumerate(SEED_URLS, 1):
            page_num = 1
            consecutive_empty_pages = 0
            product_review_count = 0  # <--- HERE: Resets to 0 for every new product
            base_clean_url = url.split('&page=')[0] if '&page=' in url else url

            while True:
                separator = "&" if "?" in base_clean_url else "?"
                current_url = f"{base_clean_url}{separator}page={page_num}"

                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = await context.new_page()
                await page.route("**/*", block_resources)

                try:
                    await asyncio.sleep(random.uniform(2.0, 3.5))
                    response = await page.goto(current_url, timeout=30000)

                    if response and (response.status in [403, 429] or "Access Denied" in await page.title()):
                        trigger_ip_rotation()
                        await context.close()
                        continue 

                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    await asyncio.sleep(1)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(1.5)

                    extracted_reviews = await page.evaluate('''() => {
                        const results = [];
                        const elements = Array.from(document.querySelectorAll('div')).filter(el => 
                            el.innerText && el.innerText.includes('Verified Purchase') && el.innerText.length > 30
                        );

                        for (const el of elements) {
                            const lines = el.innerText.split('\\n').map(l => l.trim()).filter(l => l.length > 0);
                            
                            let rating = 4.0;
                            for (const line of lines) {
                                const m = line.match(/^([1-5])(\\.[0-9])?$/);
                                if (m) {
                                    rating = parseFloat(m[0]);
                                    break;
                                }
                            }

                            const bodyCandidates = lines.filter(l => 
                                !l.includes('Verified Purchase') && 
                                !l.includes('Review for:') && 
                                !l.includes('READ MORE') && 
                                !/^\\d+$/.test(l) &&
                                l.length >= 20
                            );

                            if (bodyCandidates.length > 0) {
                                const reviewText = bodyCandidates.sort((a, b) => b.length - a.length)[0];
                                results.push({
                                    rating: rating,
                                    text: reviewText
                                });
                            }
                        }
                        return results;
                    }''')

                    batch_data = []
                    for item in extracted_reviews:
                        text = item["text"].replace("READ MORE", "").strip()
                        
                        if len(text) < 20:
                            continue

                        text_hash = get_text_hash(text)
                        if text_hash in seen_hashes:
                            continue
                        seen_hashes.add(text_hash)

                        rating_val = float(item["rating"])
                        sentiment = 0 if rating_val <= 3.0 else 1

                        batch_data.append({
                            "category": CURRENT_CATEGORY,
                            "product_url": url,
                            "rating": rating_val,
                            "sentiment": sentiment,
                            "review_text": text
                        })

                    if batch_data:
                        df_batch = pd.DataFrame(batch_data)
                        df_batch.to_csv(DATA_FILE, mode='a', header=False, index=False)
                        total_rows += len(batch_data)
                        product_review_count += len(batch_data)  # <--- HERE: Increments count
                        consecutive_empty_pages = 0
                    else:
                        consecutive_empty_pages += 1

                    print(f"Product {product_idx}/{len(SEED_URLS)} | Page {page_num} | Extracted: {len(batch_data)} | Product Total: {product_review_count}/{MAX_REVIEWS_PER_PRODUCT} | Corpus Total: {total_rows}")

                    # Stop condition 1: Reached max review cap for this product
                    if product_review_count >= MAX_REVIEWS_PER_PRODUCT:  # <--- HERE: Checks cap
                        print(f"Reached cap of {MAX_REVIEWS_PER_PRODUCT} reviews for Product {product_idx}. Moving to next product.")
                        await context.close()
                        break

                    # Stop condition 2: Reached end of reviews for this product
                    if consecutive_empty_pages >= 2:
                        print(f"Reached end of reviews for Product {product_idx}.")
                        await context.close()
                        break

                    page_num += 1

                except Exception as e:
                    print(f"Error on Product {product_idx}, Page {page_num}: {str(e)}")
                    trigger_ip_rotation()
                
                finally:
                    await context.close()

        await browser.close()
        print(f"\nScraping complete. Total unique valid reviews collected: {total_rows}")

if __name__ == "__main__":
    asyncio.run(scrape_flipkart())