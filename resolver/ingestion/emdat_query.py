"""GraphQL query fragments for the EM-DAT public API."""

from __future__ import annotations

from textwrap import dedent

EMDAT_PA_QUERY = dedent(
    """
    query PublicEmdat(
      $iso: [String!]
      $from: Int!
      $to: Int!
      $classif: [String!]!
    ) {
      api_version
      public_emdat(
        filters: {
          iso: $iso
          from: $from
          to: $to
          classif: $classif
          include_hist: false
        }
        cursor: { limit: -1 }
      ) {
        total_available
        info {
          timestamp
          version
          filters
          cursor
        }
        data {
          disno
          classif_key
          type
          subtype
          iso
          country
          start_year
          start_month
          start_day
          end_year
          end_month
          end_day
          total_affected
          entry_date
          last_update
        }
      }
    }
    """
)


EMDAT_METADATA_QUERY = dedent(
    """
    query PublicEmdatMetadata(
      $iso: [String!]
      $from: Int!
      $to: Int!
      $classif: [String!]!
    ) {
      api_version
      public_emdat(
        filters: {
          iso: $iso
          from: $from
          to: $to
          classif: $classif
          include_hist: false
        }
        cursor: { limit: 1 }
      ) {
        total_available
        info {
          timestamp
          version
        }
      }
    }
    """
)


def apply_limit_override(query: str, limit: int | None) -> str:
    """Return *query* with the cursor limit overridden when *limit* is provided."""

    if limit is None:
        return query
    # The default query pins the cursor limit to ``-1``. When a specific
    # ``limit`` value is requested we replace that literal with the caller's
    # override. This keeps the base query readable while avoiding a second,
    # format-heavy template string.
    return query.replace("limit: -1", f"limit: {int(limit)}")


__all__ = ["EMDAT_PA_QUERY", "EMDAT_METADATA_QUERY", "apply_limit_override"]
