SELECT
  FC_COD_ESTADISTICO AS SKU_A,
  FC_COD_ESTADISTICO_TIA AS SKU_B,
  FC_FECHA AS DATE_INDEX,
  FC_STOCK_INICIAL_UNID AS STOCK_INITIAL,
  FC_CONSUMO_REAL_UNID AS SOLD,
  FC_DESPACHO_UNID AS REPLENISHED,
  FC_BOTADAS_UNID AS TRASHED,
  FC_ENTRADAS_UNID AS ENTRIES,
  FC_AJUSTES_UNID AS ADJUSTMENTS,
  FC_STOCK_UNID AS STOCK_FINAL,
  FC_UNID_STOCK_MAXIMO AS STOCK_LIMIT,
  FC_ES_OFERTA AS IS_ON_SALE,
  FC_PVP AS UNIT_PRICE,
  FC_PRECIO_SIN_IVA AS UNIT_UTILITY,
  FC_COSTO_PRO_SIN_IVA AS UNIT_COST
FROM DW_IPRODUCT_FACT
WHERE FC_COD_SUCURSAL = [STORE-ID] AND
      SKU_B IN [PRESELECTED_SKUS]
ORDER BY SKU_B, DATE_INDEX;
