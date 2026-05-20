


export type SourceType = "bluesky" | "mastodon";


export interface NewsItem {

  id: string;
  source: SourceType;


  url: string;


  title: string | null;


  text: string;


  author: string | null;


  author_handle: string | null;


  created_at: string | null;


  likes_count: number | null;
  reposts_count: number | null;
  replies_count: number | null;


  language: string | null;
}


export interface ClassifiedPost extends NewsItem {
  classification: {
    label: "FAKE" | "REAL" | "UNCERTAIN";
    confidence: number;
    probability: number | null;
    reason?: string;
  } | null;
}


export interface SourceSearchParams {
  query: string;
  sources: string;
  limit: number;
}


export interface SourceRecentParams {
  sources: string;
  limit: number;
}


export interface SourceFetchUrlRequest {
  url: string;
}


export interface SourceSearchResponse {
  posts: NewsItem[];
  total: number;
  sources_used: SourceType[];
}
