"""Split CRAG's 'open' domain into finer subclasses.

Reads crag_domain_query.json. Keeps the original 4 specialized domains
(finance/movie/music/sports) and re-labels each 'open' query into one of:
    greetings, math_code, food, celebrities, relationship_psychology
Unmatched 'open' queries stay as 'open' (fallback).

Writes a new file where each item has a new `class` field. The original
`domain` field is preserved for traceability.
"""

import json
import re
from collections import Counter
from pathlib import Path

INPUT = Path("/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed/crag_domain_query.json")
OUTPUT = Path(
    "/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed/crag_domain_query_subclass.json"
)

# Priority order matters — first match wins.
SUBCATS = [
    # 1. Greetings / chit-chat. Anchored to keep it small.
    (
        "greetings",
        r"^\s*(hi|hello|hey|hola|good (morning|afternoon|evening|night)|greetings|thanks|thank you|how are you|what's up|whats up)\b",
    ),
    # 2. Food / cooking.
    (
        "food",
        r"\b(food|foods|recipe|recipes|cook|cooks|cooked|cooking|bake|baked|baking|cake|bread|cheese|sugar|salt|chocolate|coffee|tea|wine|beer|liquor|cocktail|restaurant|restaurants|meal|breakfast|lunch|dinner|sauce|salad|soup|cuisine|cuisines|ingredient|ingredients|kitchen|fruit|fruits|vegetable|vegetables|meat|fish|truffle|spice|spices|chef|chefs|tablespoon|teaspoon|cup of|calorie|calories|edible|eat|eats|ate|drink|drinks|beverage|beverages|dish|dishes|mcdonald|kfc|burger|pizza|sushi|pasta|noodle|rice|seafood|dairy|gluten|vegan|vegetarian)\b",
    ),
    # 3. Math / code. Strict programming or arithmetic vocab.
    (
        "math_code",
        r"\b(math|maths|mathematics|equation|equations|integral|derivative|algebra|geometry|calculus|probability|statistic|statistics|matrix|vector|theorem|prime number|fraction|fractions|formula|formulas|algorithm|algorithms|code|coding|program|programming|python|java|javascript|c\+\+|html|css|sql|software|developer|programmer|debug|compile|compiler|script|django|react|node\.js|typescript|regex|terminal|shell|linux|kernel|api|github|stack overflow|computer science|big o|sorting algorithm|hash|binary|recursion|loop)\b",
    ),
    # 4. Gaming — video games, publishers, consoles.
    (
        "gaming",
        r"\b(video game|video games|videogame|videogames|game publisher|game developer|game studio|playstation|xbox|nintendo|switch|wii|game ?boy|sega|atari|steam (deck|game)|pc gaming|esports|e-sport|mmorpg|rpg|fps|moba|battle royale|minecraft|fortnite|roblox|league of legends|call of duty|grand theft auto|gta|world of warcraft|elden ring|zelda|mario|pokemon|pokémon|metroid|halo|tetris|pac[- ]?man|donkey kong|sonic the hedgehog|dig dug|robocop: rogue city|the wreck \(video game\))\b",
    ),
    # 5. Art / history — painters, art works, dynasties, monuments, historical figures.
    (
        "art_history",
        r"\b(painter|painters|painting|paintings|sculpture|sculptures|sculptor|art exhibit|art exhibition|gallery|museum|museums|louvre|metropolitan museum|guggenheim|moma|renaissance|baroque|impressionism|impressionist|cubism|surrealism|surrealist|abstract art|modern art|contemporary art|van gogh|picasso|monet|rembrandt|da vinci|leonardo da vinci|michelangelo|rothko|escher|frida kahlo|warhol|dynasty|dynasties|empire|empires|emperor|emperors|pharaoh|pharaohs|medieval|ancient|antiquity|world war|wwi|wwii|civil war|cold war|revolution|battle of|napoleon|caesar|cleopatra|alexander the great|monarchy|monarch|monarchs|monument|monuments|cathedral|cathedrals|mosque|mosques|temple|temples|palace|palaces|castle|castles|pyramid|pyramids)\b",
    ),
    # 6. Business / companies — non-finance corporate questions (employees, owners, locations).
    (
        "business_companies",
        r"\b(walmart|aldi|costco|amazon\.com|amazon\.com inc|kroger|target corporation|home depot|lowe's|h-?e-?b|heb (store|locations)|whole foods|trader joe's|7[- ]?eleven|publix|safeway|ikea|starbucks|mcdonalds|mcdonald's|chipotle|wholesale store|grocery (store|chain)|retail chain|retailer|retailers|company headquarter|headquartered in|founded in|founded by|founder of|owner of|owners of|owned by|publisher of|publishing company|holding company|subsidiary|subsidiaries|conglomerate|conglomerates|fortune 500|fortune 100|how (much|many) employees|number of employees|employee count|ticker symbol)\b",
    ),
    # 7. Geography / places — countries, cities, lakes, mountains, populations, dimensions.
    (
        "geography_places",
        r"\b(country|countries|capital city|capital of|continent|continents|city|cities|town|towns|village|villages|state of|state's|province|provinces|region|regions|county|counties|river|rivers|lake|lakes|ocean|oceans|sea|seas|gulf|bay|strait|peninsula|mountain|mountains|peak|peaks|volcano|volcanoes|island|islands|desert|deserts|forest|forests|national park|map of|border|borders|bordering|population of|populated|inhabitants|how many people live|sq km|square kilometer|square mile|square miles|area code|area codes|time zone|time zones|latitude|longitude|coordinate|coordinates|northern hemisphere|southern hemisphere|northern africa|southern africa|eastern europe|western europe|north america|south america|central america|middle east|southeast asia|south asia|east asia|africa|europe|asia|oceania|antarctica|gdp of|currency of|currency used|spoken in|official language|language(s)? of|dam|dams|nuclear power plant)\b",
    ),
    # 8. Relationship / psychology. Broader than celeb-relationships — keep psych signals.
    (
        "relationship_psychology",
        r"\b(psychology|psychological|psychiatry|psychiatrist|therapist|therapy|counseling|mental health|anxiety|depression|stress|stressful|emotion|emotions|emotional|feeling|feelings|mindfulness|self[- ]?esteem|self[- ]?help|loneliness|happiness|sadness|grief|trauma|ptsd|bipolar|adhd|autism|relationship advice|breakup|divorce|dating advice|love languages|attachment style|friendship|friendships)\b",
    ),
    # 9. Celebrities. Common bio signal words + "who is/was" + actor/singer/etc roles.
    (
        "celebrities",
        r"\b(actor|actors|actress|actresses|singer|singers|songwriter|musician|musicians|rapper|rappers|director|directors|producer|producers|filmmaker|filmmakers|athlete|athletes|player|players|coach|coaches|ceo|president|presidents|king|queen|emperor|emperors|prince|princess|pope|celebrity|celebrities|star|stars|husband|husbands|wife|wives|spouse|spouses|partner|partners|girlfriend|boyfriend|married|marriage|dating|divorced|parents|parent|mother|father|sister|brother|son|daughter|child|children|kids|born in|died in|date of birth|date of death|net worth|biography|biographical|famous for|known for|nominated for|grammy|oscar|emmy|tony award|nobel prize|net worth|followers|subscribers|tweet|tweeted|instagram|tiktok|youtube channel|first lady|vice president|prime minister|monarch)\b",
    ),
]
PATTERNS = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in SUBCATS]


def subclassify_open(text: str) -> str:
    if not isinstance(text, str):
        return "open"
    for name, pat in PATTERNS:
        if pat.search(text):
            return name
    return "open"  # fallback for unmatched open queries


def main() -> None:
    with INPUT.open() as f:
        data = json.load(f)

    out = []
    for item in data:
        new_item = dict(item)
        domain = item.get("domain", "")
        if domain == "open":
            new_item["class"] = subclassify_open(item.get("query", ""))
        else:
            new_item["class"] = domain
        out.append(new_item)

    with OUTPUT.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"in={len(data)} out={len(out)} -> {OUTPUT}")
    print("class distribution:")
    for k, v in Counter(x["class"] for x in out).most_common():
        print(f"  {k:28s} {v}")
    print("\nopen-only subclass breakdown:")
    opens = [x for x in out if x["domain"] == "open"]
    for k, v in Counter(x["class"] for x in opens).most_common():
        print(f"  {k:28s} {v}/{len(opens)}")


if __name__ == "__main__":
    main()
