#include "cuda_syntax_highlighter.h"


CUDASyntaxHighlighter::CUDASyntaxHighlighter(QTextDocument *parent)
    : QSyntaxHighlighter(parent)
{
  const QString keyword_patterns[] = {
      QStringLiteral("\\bfor\\b"),    QStringLiteral("\\bif\\b"),    QStringLiteral("\\band\\b"),
      QStringLiteral("\\bbreak\\b"),  QStringLiteral("\\bclass\\b"), QStringLiteral("\\bcontinue\\b"),
      QStringLiteral("\\belse\\b"),   QStringLiteral("\\bnot\\b"),   QStringLiteral("\\bor\\b"),
      QStringLiteral("\\breturn\\b"), QStringLiteral("\\btry\\b"),   QStringLiteral("\\bwhile\\b"),
      QStringLiteral("\\btrue\\b"),   QStringLiteral("\\bfalse\\b"), QStringLiteral("\\bprintf\\b"),

      QStringLiteral("\\__global__\\b"),
      QStringLiteral("\\threadIdx\\b"),
      QStringLiteral("\\blockIdx\\b"),
      QStringLiteral("\\blockDim\\b"),
  };

  const QString numbers_patterns[] = {
      QStringLiteral("(\\d+)")
  };

  // keyword_format.setForeground(QBrush(QColor("#be93f9"))); // __global__
  keyword_format.setForeground(QBrush(QColor("#ff78b7"))); // for
  numbers_format.setForeground(QBrush(QColor("#ff78b7")));

  HighlightingRule rule;

  for (auto &keyword: keyword_patterns)
  {
    rule.format = keyword_format;
    rule.pattern = QRegularExpression (keyword);
    highlighting_rules.push_back (rule);
  }

  for (auto &num: numbers_patterns)
  {
    rule.format = numbers_format;
    rule.pattern = QRegularExpression (num);
    highlighting_rules.push_back (rule);
  }
}

void CUDASyntaxHighlighter::highlightBlock(const QString &text)
{
  for (const HighlightingRule &rule : qAsConst(highlighting_rules)) {
    QRegularExpressionMatchIterator matchIterator = rule.pattern.globalMatch(text);
    while (matchIterator.hasNext()) {
      QRegularExpressionMatch match = matchIterator.next();
      setFormat(match.capturedStart(), match.capturedLength(), rule.format);
    }
  }
  setCurrentBlockState(0);
}