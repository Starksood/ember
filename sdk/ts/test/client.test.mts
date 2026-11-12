import assert from "node:assert/strict";
import test from "node:test";
import { EmberApiError, EmberClient } from "../src/index.ts";

test("EmberClient sends Bearer token, X-Ember-Tenant, and surfaces typed EmberApiError", async () => {
  const seen: { url: string; method: string; auth?: string; tenant?: string; body?: string }[] = [];

  const mockFetch: typeof fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.toString();
    const hdrs = (init?.headers ?? {}) as Record<string, string>;
    seen.push({
      url,
      method: init?.method ?? "GET",
      auth: hdrs.Authorization,
      tenant: hdrs["X-Ember-Tenant"],
      body: typeof init?.body === "string" ? init.body : undefined,
    });
    if (url.endsWith("/health")) {
      return new Response(
        JSON.stringify({
          status: "ok",
          version: "0.3.0",
          tenant_id: "acme",
          keys_separated_from_ledger: true,
          encrypt_vendor_payload: false,
        }),
        { status: 200 },
      );
    }
    if (url.endsWith("/v1/remember")) {
      return new Response(
        JSON.stringify({
          error: "TombstonedSubjectError",
          detail: "subject erased",
        }),
        { status: 403 },
      );
    }
    if (url.includes("/v1/coverage")) {
      return new Response(
        JSON.stringify({
          tenant_id: "acme",
          subject_id: "alice",
          verdict: "bounded_with_warnings",
          erasure_certifiable: true,
          summary: "Bounded with warnings",
          ledger_chain_ok: true,
          source_count: 1,
          sources_linked_count: 1,
          orphan_source_ids: [],
          source_coverage_rate: 1.0,
          live_item_count: 2,
          attributed_item_count: 1,
          unattributed_item_count: 1,
          unattributed_item_ids: ["item_unattr"],
          traced_item_count: 0,
          derived_item_count: 0,
          derived_dependency_count: 0,
          lineage_capture_rate: 0.5,
          verified_stores: [],
          unverified_locations: [],
          pending_erasure_count: 0,
          pending_erasures: [],
          recommendations: [],
        }),
        { status: 200 },
      );
    }
    if (url.endsWith("/v1/reconcile")) {
      return new Response(
        JSON.stringify({
          attempted: 1,
          all_clean: true,
          completed_ids: ["era_1"],
          still_failing: [],
          certificate_ids: ["cert_1"],
        }),
        { status: 200 },
      );
    }
    return new Response(JSON.stringify({ query: "q", abstained: true, filtered_orphans: 0, hits: [] }), { status: 200 });
  };

  const client = new EmberClient({
    baseUrl: "http://127.0.0.1:8420/",
    bearerToken: "tok_secret",
    tenantId: "acme",
    fetchFn: mockFetch,
  });

  const h = await client.health();
  assert.equal(h.status, "ok");
  assert.equal(seen[0].auth, undefined);

  const s = await client.search({ query: "q", limit: 3 });
  assert.equal(s.abstained, true);
  assert.equal(seen[1].auth, "Bearer tok_secret");
  assert.equal(seen[1].tenant, "acme");

  await assert.rejects(
    () => client.remember({ text: "hello", subjects: ["alice"] }),
    (err: unknown) => {
      assert.ok(err instanceof EmberApiError);
      assert.equal(err.status, 403);
      assert.equal(err.code, "TombstonedSubjectError");
      return true;
    },
  );

  const cov = await client.coverage({ subjectId: "alice", unverifiedLocations: ["s3_snapshot"] });
  assert.equal(cov.verdict, "bounded_with_warnings");
  assert.equal(cov.unattributed_item_count, 1);

  const rec = await client.reconcile();
  assert.equal(rec.all_clean, true);
  assert.deepEqual(rec.certificate_ids, ["cert_1"]);
});

const liveUrl = process.env.EMBER_LIVE_TEST_URL;
const liveToken = process.env.EMBER_LIVE_TEST_TOKEN;

if (liveUrl) {
  test("EmberClient live round-trip against Python sidecar", async () => {
    const client = new EmberClient({
      baseUrl: liveUrl,
      bearerToken: liveToken,
      tenantId: process.env.EMBER_LIVE_TEST_TENANT ?? "default",
    });

    const health = await client.health();
    assert.equal(health.status, "ok");

    const src = await client.registerSource({
      source_id: "ts_doc",
      text: "Maya takes warfarin daily, and Leo plays violin.",
      span_mode: "clause",
      origin: "node_e2e",
    });
    assert.ok(src.merkle_root.length === 64);

    await client.remember({
      text: "Maya takes warfarin daily",
      subjects: ["maya"],
      source_id: "ts_doc",
    });
    await client.remember({
      text: "Leo plays violin",
      subjects: ["leo"],
      source_id: "ts_doc",
    });

    const cert = await client.erase({ subject_id: "maya", probes: ["warfarin", "Maya"] });
    assert.equal(cert.key_destroyed, true);
    assert.ok(cert.signature.startsWith("ed25519:"));

    const mayaAfter = await client.search({ query: "warfarin" });
    assert.equal(mayaAfter.abstained, true);

    const cov = await client.coverage({ subjectId: "leo" });
    assert.equal(cov.verdict, "complete");
    assert.equal(cov.ledger_chain_ok, true);

    const rec = await client.reconcile();
    assert.equal(rec.all_clean, true);

    const leoAfter = await client.search({ query: "violin" });
    assert.equal(leoAfter.abstained, false);
    assert.equal(leoAfter.hits[0].receipt_verified, true);
  });
}
