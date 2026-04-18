import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  Modal,
  ScrollView,
  Dimensions,
  SafeAreaView,
} from "react-native";
import { useTheme } from "../context/ThemeContext";
import CoverageBars from "./CoverageBars";

const { width: W } = Dimensions.get("window");
const CELL_SIZE = (W - 16 * 2 - 8) / 2; // 2 columns, 16 padding each side, 8 gap

const RIPENESS_COLORS = {
  fully_ripened: "#FF5078",
  half_ripened: "#5078FF",
  green: "#50C850",
};

const RIPENESS_LABELS = {
  fully_ripened: "Fully Ripened",
  half_ripened: "Half Ripened",
  green: "Green",
};

function getDominant(coverage) {
  if (!coverage) return null;
  const groups = {
    fully_ripened:
      (coverage.b_fully_ripened ?? 0) + (coverage.l_fully_ripened ?? 0),
    half_ripened:
      (coverage.b_half_ripened ?? 0) + (coverage.l_half_ripened ?? 0),
    green: (coverage.b_green ?? 0) + (coverage.l_green ?? 0),
  };
  const total = Object.values(groups).reduce((a, b) => a + b, 0);
  if (total < 0.5) return null;
  const [key, val] = Object.entries(groups).sort((a, b) => b[1] - a[1])[0];
  return { key, pct: ((val / total) * 100).toFixed(0) };
}

function getTomatoTotal(coverage) {
  if (!coverage) return 0;
  return (
    (coverage.b_fully_ripened ?? 0) +
    (coverage.b_half_ripened ?? 0) +
    (coverage.b_green ?? 0) +
    (coverage.l_fully_ripened ?? 0) +
    (coverage.l_half_ripened ?? 0) +
    (coverage.l_green ?? 0)
  );
}

// ── Single thumbnail cell ─────────────────────────────────────────────────────
function GridCell({ item, index, onPress, t }) {
  const dominant = getDominant(item.coverage);
  const total = getTomatoTotal(item.coverage);
  const color = dominant ? RIPENESS_COLORS[dominant.key] : t.borderMid;

  return (
    <TouchableOpacity
      style={[cs.cell, { backgroundColor: t.card, borderColor: t.border }]}
      onPress={() => onPress(item, index)}
      activeOpacity={0.8}
    >
      {/* Overlay image */}
      {item.overlay_b64 ? (
        <Image
          source={{ uri: `data:image/jpeg;base64,${item.overlay_b64}` }}
          style={cs.cellImage}
          resizeMode="cover"
        />
      ) : (
        <View
          style={[
            cs.cellImage,
            cs.cellPlaceholder,
            { backgroundColor: t.cardAlt },
          ]}
        >
          <Text style={{ fontSize: 28 }}>🍅</Text>
        </View>
      )}

      {/* Dominant badge */}
      {dominant ? (
        <View
          style={[
            cs.badge,
            { backgroundColor: color + "22", borderColor: color + "55" },
          ]}
        >
          <View style={[cs.badgeDot, { backgroundColor: color }]} />
          <Text style={[cs.badgeText, { color }]}>{dominant.pct}%</Text>
        </View>
      ) : (
        <View
          style={[
            cs.badge,
            { backgroundColor: t.cardAlt, borderColor: t.border },
          ]}
        >
          <Text style={[cs.badgeText, { color: t.textTertiary }]}>—</Text>
        </View>
      )}

      {/* Footer */}
      <View style={[cs.cellFooter, { backgroundColor: t.card }]}>
        {dominant && (
          <Text style={[cs.cellLabel, { color }]} numberOfLines={1}>
            {RIPENESS_LABELS[dominant.key]}
          </Text>
        )}
        <Text style={[cs.cellSub, { color: t.textTertiary }]} numberOfLines={1}>
          {total > 0 ? `${total.toFixed(1)}% tomato` : "No tomatoes"}
        </Text>
        {item.filename && (
          <Text
            style={[cs.cellFilename, { color: t.textMuted }]}
            numberOfLines={1}
          >
            {item.filename}
          </Text>
        )}
      </View>
    </TouchableOpacity>
  );
}

const cs = StyleSheet.create({
  cell: {
    width: CELL_SIZE,
    borderRadius: 12,
    borderWidth: 1,
    overflow: "hidden",
    marginBottom: 8,
  },
  cellImage: {
    width: "100%",
    height: CELL_SIZE * 0.75,
  },
  cellPlaceholder: {
    alignItems: "center",
    justifyContent: "center",
  },
  badge: {
    position: "absolute",
    top: 8,
    right: 8,
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 20,
    borderWidth: 1,
  },
  badgeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  badgeText: {
    fontSize: 10,
    fontWeight: "700",
  },
  cellFooter: {
    padding: 8,
    gap: 2,
  },
  cellLabel: {
    fontSize: 11,
    fontWeight: "700",
  },
  cellSub: {
    fontSize: 10,
  },
  cellFilename: {
    fontSize: 9,
    marginTop: 1,
  },
});

// ── Detail modal ──────────────────────────────────────────────────────────────
function DetailModal({ item, index, total, onClose, onPrev, onNext, t }) {
  if (!item) return null;
  const dominant = getDominant(item.coverage);
  const tomatoTotal = getTomatoTotal(item.coverage);
  const color = dominant ? RIPENESS_COLORS[dominant.key] : t.accent;

  return (
    <Modal visible animationType="slide" transparent onRequestClose={onClose}>
      <View style={ms.overlay}>
        <TouchableOpacity
          style={ms.backdrop}
          onPress={onClose}
          activeOpacity={1}
        />
        <View
          style={[ms.sheet, { backgroundColor: t.card, borderColor: t.border }]}
        >
          {/* Image */}
          {item.overlay_b64 && (
            <Image
              source={{ uri: `data:image/jpeg;base64,${item.overlay_b64}` }}
              style={ms.image}
              resizeMode="cover"
            />
          )}

          {/* Nav arrows over image */}
          <View style={ms.navRow} pointerEvents="box-none">
            <TouchableOpacity
              style={[
                ms.navBtn,
                {
                  backgroundColor: t.hudBg ?? "rgba(0,0,0,0.5)",
                  opacity: index === 0 ? 0.3 : 1,
                },
              ]}
              onPress={onPrev}
              disabled={index === 0}
            >
              <Text style={ms.navArrow}>‹</Text>
            </TouchableOpacity>
            <View
              style={[
                ms.navCounter,
                { backgroundColor: t.hudBg ?? "rgba(0,0,0,0.5)" },
              ]}
            >
              <Text style={ms.navCounterText}>
                {index + 1} / {total}
              </Text>
            </View>
            <TouchableOpacity
              style={[
                ms.navBtn,
                {
                  backgroundColor: t.hudBg ?? "rgba(0,0,0,0.5)",
                  opacity: index === total - 1 ? 0.3 : 1,
                },
              ]}
              onPress={onNext}
              disabled={index === total - 1}
            >
              <Text style={ms.navArrow}>›</Text>
            </TouchableOpacity>
          </View>

          {/* Body */}
          <ScrollView
            contentContainerStyle={ms.body}
            showsVerticalScrollIndicator={false}
          >
            {/* Header */}
            <View style={ms.header}>
              <View>
                {dominant && (
                  <View
                    style={[
                      ms.dominantBadge,
                      {
                        backgroundColor: color + "18",
                        borderColor: color + "44",
                      },
                    ]}
                  >
                    <View
                      style={[ms.dominantDot, { backgroundColor: color }]}
                    />
                    <Text style={[ms.dominantLabel, { color }]}>
                      {RIPENESS_LABELS[dominant.key]}
                    </Text>
                    <Text style={[ms.dominantPct, { color: t.textPrimary }]}>
                      {dominant.pct}%
                    </Text>
                  </View>
                )}
                {item.filename && (
                  <Text style={[ms.filename, { color: t.textTertiary }]}>
                    {item.filename}
                  </Text>
                )}
              </View>
              <TouchableOpacity
                style={[ms.closeBtn, { backgroundColor: t.cardAlt }]}
                onPress={onClose}
              >
                <Text style={[ms.closeText, { color: t.textPrimary }]}>✕</Text>
              </TouchableOpacity>
            </View>

            {/* Coverage bars */}
            <CoverageBars
              coverage={item.coverage}
              confidence={item.confidence}
            />

            {/* Total */}
            <View style={[ms.totalRow, { borderTopColor: t.border }]}>
              <Text style={[ms.totalLabel, { color: t.textSec }]}>
                Total tomato coverage
              </Text>
              <Text style={[ms.totalVal, { color: t.textPrimary }]}>
                {tomatoTotal.toFixed(1)}%
              </Text>
            </View>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

const ms = StyleSheet.create({
  overlay: { flex: 1, justifyContent: "flex-end" },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.6)",
  },
  sheet: {
    borderTopLeftRadius: 22,
    borderTopRightRadius: 22,
    borderWidth: 1,
    borderBottomWidth: 0,
    overflow: "hidden",
    maxHeight: "90%",
  },
  image: { width: "100%", height: 220 },
  navRow: {
    position: "absolute",
    top: 220 - 44,
    left: 0,
    right: 0,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 12,
  },
  navBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
  },
  navArrow: { color: "#fff", fontSize: 22, fontWeight: "700" },
  navCounter: {
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 20,
  },
  navCounterText: { color: "#fff", fontSize: 12, fontWeight: "600" },
  body: { padding: 16, gap: 12, paddingBottom: 32 },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
  },
  dominantBadge: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    gap: 7,
    marginBottom: 6,
  },
  dominantDot: { width: 8, height: 8, borderRadius: 4 },
  dominantLabel: { fontSize: 13, fontWeight: "700" },
  dominantPct: { fontSize: 13, fontWeight: "700" },
  filename: { fontSize: 11, marginTop: 2 },
  closeBtn: {
    width: 32,
    height: 32,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  closeText: { fontSize: 14, fontWeight: "600" },
  totalRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingTop: 12,
    borderTopWidth: 1,
    marginTop: 4,
  },
  totalLabel: { fontSize: 13 },
  totalVal: { fontSize: 13, fontWeight: "700" },
});

// ── Main BatchGrid ────────────────────────────────────────────────────────────
export default function BatchGrid({ results }) {
  const { theme: t } = useTheme();
  const [selectedIndex, setSelectedIndex] = useState(null);

  if (!results || results.length === 0) return null;

  const selected = selectedIndex !== null ? results[selectedIndex] : null;

  return (
    <View style={{ flex: 1 }}>
      {/* Summary bar */}
      <View
        style={[
          gs.summaryBar,
          { backgroundColor: t.card, borderColor: t.border },
        ]}
      >
        <Text style={[gs.summaryCount, { color: t.textPrimary }]}>
          {results.length} image{results.length > 1 ? "s" : ""} analysed
        </Text>
        <Text style={[gs.summaryHint, { color: t.textTertiary }]}>
          Tap to inspect
        </Text>
      </View>

      {/* Grid */}
      <FlatList
        data={results}
        keyExtractor={(_, i) => i.toString()}
        numColumns={2}
        columnWrapperStyle={gs.row}
        contentContainerStyle={gs.grid}
        showsVerticalScrollIndicator={false}
        renderItem={({ item, index }) => (
          <GridCell
            item={item}
            index={index}
            onPress={(_, i) => setSelectedIndex(i)}
            t={t}
          />
        )}
      />

      {/* Detail modal */}
      {selected && (
        <DetailModal
          item={selected}
          index={selectedIndex}
          total={results.length}
          onClose={() => setSelectedIndex(null)}
          onPrev={() => setSelectedIndex((i) => Math.max(0, i - 1))}
          onNext={() =>
            setSelectedIndex((i) => Math.min(results.length - 1, i + 1))
          }
          t={t}
        />
      )}
    </View>
  );
}

const gs = StyleSheet.create({
  summaryBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginHorizontal: 16,
    marginBottom: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
  },
  summaryCount: { fontSize: 13, fontWeight: "700" },
  summaryHint: { fontSize: 11 },
  grid: { paddingHorizontal: 16, paddingBottom: 24 },
  row: { justifyContent: "space-between" },
});
