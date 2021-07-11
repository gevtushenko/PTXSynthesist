#pragma once

// Qt
#include <QSyntaxHighlighter> // Required for inheritance

class SyntaxStyle;

class StyleSyntaxHighlighter : public QSyntaxHighlighter
{
public:

    /**
     * @brief Constructor.
     * @param document Pointer to text document.
     */
    explicit StyleSyntaxHighlighter(QTextDocument* document=nullptr);

    // Disable copying
    StyleSyntaxHighlighter(const StyleSyntaxHighlighter&) = delete;
    StyleSyntaxHighlighter& operator=(const StyleSyntaxHighlighter&) = delete;

    /**
     * @brief Method for setting syntax style.
     * @param style Pointer to syntax style.
     */
    void setSyntaxStyle(SyntaxStyle* style);

    /**
     * @brief Method for getting syntax style.
     * @return Pointer to syntax style. May be nullptr.
     */
    SyntaxStyle* syntaxStyle() const;

private:
    SyntaxStyle* m_syntaxStyle;
};

