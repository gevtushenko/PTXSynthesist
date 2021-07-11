#pragma once

// CodeEditor
#include <style_syntax_highlighter.hpp> // Required for inheritance
#include <highlight_rule.hpp>

// Qt
#include <QRegularExpression>
#include <QVector>

class SyntaxStyle;

/**
 * @brief Class, that describes C++ code
 * highlighter.
 */
class PTXHighlighter : public StyleSyntaxHighlighter
{
    Q_OBJECT
public:

    /**
     * @brief Constructor.
     * @param document Pointer to document.
     */
    explicit PTXHighlighter(QTextDocument* document=nullptr);

protected:
    void highlightBlock(const QString& text) override;

private:

    QVector<HighlightRule> m_highlightRules;

    QRegularExpression m_includePattern;
    QRegularExpression m_functionPattern;
    QRegularExpression m_defTypePattern;

    QRegularExpression m_commentStartPattern;
    QRegularExpression m_commentEndPattern;
};

