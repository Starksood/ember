export type ChunkMode = "sentence" | "line" | "whole";
export type SpanMode = "word" | "clause" | null;

export interface EmberClientOptions {
  baseUrl: string;
  bearerToken?: string;
  tenantId?: string;
  fetchFn?: typeof fetch;
}

export interface RegisterSourceRequest {
  source_id: string;
  text: string;
  mode?: ChunkMode;
  span_mode?: SpanMode;
  origin?: string;
  origin_kind?: string;
  supplied_by?: string;
  validated_by?: string;
  event_id?: string;
}

export interface SourceRecord {
  source_id: string;
  merkle_root: string;
  flat_doc_hash: string;
  byte_length: number;
  leaf_count: number;
  chunk_mode: string;
  ingested_at: string;
  origin: string;
  origin_kind: string;
  supplied_by: string;
  validated_by: string;
}

export interface RegisterSourceResponse {
  source: SourceRecord;
  merkle_root: string;
}

export interface RememberRequest {
  text: string;
  subjects: string[];
  source_id?: string;
  quote?: string;
  leaf_index?: number;
  derived_from?: string[];
  aliases?: string[];
  metadata?: Record<string, unknown>;
  reconsent_id?: string;
  item_id?: string;
  event_id?: string;
  auto_derive?: boolean;
  content_guard?: boolean;
}

export interface Receipt {
  source_id: string;
  quote: string;
  flat_doc_hash: string;
  byte_start: number;
  byte_end: number;
  merkle_root: string;
  leaf_index: number;
  leaf_count: number;
  proof: [string, string][];
  span_indices?: number[];
  span_count?: number;
  span_proofs?: [string, string][][];
}

export interface RememberResponse {
  item_id: string;
  event_hash: string;
  adapter_record_ids: string[];
  subjects: string[];
  derived_from: string[];
  receipt: Receipt | null;
}

export interface SearchRequest {
  query: string;
  limit?: number;
  filters?: Record<string, unknown>;
  verify_receipts?: boolean;
}

export interface QueryHit {
  item_id: string;
  record_id: string;
  text: string;
  score: number;
  subjects: string[];
  receipt_verified: boolean | null;
  receipt: Receipt | null;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  query: string;
  abstained: boolean;
  filtered_orphans: number;
  hits: QueryHit[];
}

export interface EraseRequest {
  subject_id: string;
  aliases?: string[];
  probes?: string[];
  markers?: string[];
  block_identifiers?: string[];
  auto_probes?: boolean;
}

export interface ErasureCertificate {
  certificate_id: string;
  tenant_id: string;
  subject_id: string;
  subject_tokens: string[];
  state_hash_before: string;
  state_hash_after: string;
  ledger_tail_hash: string;
  closure: Record<string, unknown>;
  probe_count: number;
  battery_all_abstained: boolean;
  cipher: string;
  key_destroyed: boolean;
  record_keys_destroyed: number;
  key_destroyed_at: string;
  issued_at: string;
  trust_boundary: Record<string, unknown>;
  anchor: { scheme: string; digest: string; token: string };
  signer_scheme: string;
  public_key: string;
  signature: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  tenant_id: string;
  keys_separated_from_ledger: boolean;
  encrypt_vendor_payload: boolean;
}

export class EmberApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: string;
  readonly payload: Record<string, unknown>;

  constructor(status: number, code: string, detail: string, payload: Record<string, unknown> = {}) {
    super(`Ember HTTP ${status} (${code}): ${detail}`);
    this.name = "EmberApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
    this.payload = payload;
  }
}

export class EmberClient {
  private readonly baseUrl: string;
  private readonly bearerToken?: string;
  private readonly tenantId?: string;
  private readonly fetchFn: typeof fetch;

  constructor(options: EmberClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.bearerToken = options.bearerToken;
    this.tenantId = options.tenantId;
    this.fetchFn = options.fetchFn ?? globalThis.fetch;
    if (typeof this.fetchFn !== "function") {
      throw new Error("EmberClient requires a Web-compatible fetch implementation");
    }
  }

  private async request<T>(method: "GET" | "POST", path: string, body?: unknown, skipAuth = false): Promise<T> {
    const headers: Record<string, string> = {
      Accept: "application/json",
    };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
    }
    if (!skipAuth && this.bearerToken) {
      headers.Authorization = `Bearer ${this.bearerToken}`;
    }
    if (!skipAuth && this.tenantId) {
      headers["X-Ember-Tenant"] = this.tenantId;
    }

    const resp = await this.fetchFn(`${this.baseUrl}${path}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const text = await resp.text();
    let parsed: Record<string, unknown> = {};
    if (text) {
      parsed = JSON.parse(text) as Record<string, unknown>;
    }
    if (!resp.ok) {
      const code = typeof parsed.error === "string" ? parsed.error : "HttpError";
      const detail = typeof parsed.detail === "string" ? parsed.detail : resp.statusText;
      throw new EmberApiError(resp.status, code, detail, parsed);
    }
    return parsed as T;
  }

  health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/health", undefined, true);
  }

  registerSource(req: RegisterSourceRequest): Promise<RegisterSourceResponse> {
    return this.request<RegisterSourceResponse>("POST", "/v1/sources", req);
  }

  remember(req: RememberRequest): Promise<RememberResponse> {
    return this.request<RememberResponse>("POST", "/v1/remember", req);
  }

  search(req: SearchRequest): Promise<SearchResponse> {
    return this.request<SearchResponse>("POST", "/v1/search", req);
  }

  erase(req: EraseRequest): Promise<ErasureCertificate> {
    return this.request<ErasureCertificate>("POST", "/v1/erase", req);
  }

  checkpoint(): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>("POST", "/v1/checkpoint", {});
  }

  exportBundle(includeLedger = true): Promise<Record<string, unknown>> {
    const q = includeLedger ? "true" : "false";
    return this.request<Record<string, unknown>>("GET", `/v1/bundle?include_ledger=${q}`);
  }
}
