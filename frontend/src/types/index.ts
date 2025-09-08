export interface Market {
  id: string
  title: string
  description: string
  endDate: string
  volume: number
  yesPrice: number
  noPrice: number
  category: string
}

export interface Trade {
  id: string
  marketId: string
  side: 'yes' | 'no'
  amount: number
  price: number
  timestamp: string
  status: 'pending' | 'completed' | 'failed'
}

export interface Agent {
  id: string
  name: string
  status: 'active' | 'inactive' | 'paused'
  balance: number
  totalTrades: number
  successRate: number
}