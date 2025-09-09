import { NextResponse } from "next/server";

const MARKET_ID = "561767"; // Demo market
const POLYMARKET_API_URL = `https://gamma-api.polymarket.com/markets/${MARKET_ID}`;

interface MarketData {
  question: string;
  icon: string;
  outcomePrices: string[];
  volume: string;
  cached?: boolean;
  cacheAge?: number;
  fetchedAt?: string;
  stale?: boolean;
  error?: string;
}

let cachedData: MarketData | null = null;
let lastFetchTime: number = 0;
const CACHE_DURATION = 5000;

export async function GET() {
  try {
    const now = Date.now();

    if (cachedData && now - lastFetchTime < CACHE_DURATION) {
      console.log("📊 Returning cached market data");
      return NextResponse.json({
        ...cachedData,
        cached: true,
        cacheAge: now - lastFetchTime,
      });
    }

    console.log("📊 Fetching fresh market data from Polymarket API");
    const response = await fetch(POLYMARKET_API_URL);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const rawData = await response.json();

    const marketData: MarketData = {
      question: rawData.question,
      icon: rawData.icon,
      outcomePrices: JSON.parse(rawData.outcomePrices),
      volume: rawData.volume,
    };

    cachedData = marketData;
    lastFetchTime = now;

    return NextResponse.json({
      ...marketData,
      cached: false,
      fetchedAt: new Date(now).toISOString(),
    });
  } catch (error) {
    console.error("❌ Error fetching market data:", error);

    if (cachedData) {
      console.log("⚠️ Returning stale cached data due to fetch error");
      return NextResponse.json({
        ...cachedData,
        cached: true,
        stale: true,
        error: "Fresh fetch failed, returning cached data",
      });
    }

    return NextResponse.json(
      { error: "Failed to fetch market data and no cached data available" },
      { status: 500 }
    );
  }
}
