//
// Created by evtus on 7/10/2021.
//

#ifndef PTXSYNTHESIST_CUDA_SYNTAX_HIGHLIGHTER_H
#define PTXSYNTHESIST_CUDA_SYNTAX_HIGHLIGHTER_H

#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>

class CUDASyntaxHighlighter : public QSyntaxHighlighter
{
  Q_OBJECT

public:
  explicit CUDASyntaxHighlighter(QTextDocument *parent);

protected:
  void highlightBlock(const QString &text) override;

private:
  struct HighlightingRule
  {
    QRegularExpression pattern;
    QTextCharFormat format;
  };

  QVector<HighlightingRule> highlighting_rules;
  QTextCharFormat keyword_format, numbers_format;
};

#endif //PTXSYNTHESIST_CUDA_SYNTAX_HIGHLIGHTER_H
