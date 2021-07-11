#pragma once

// Qt
#include <QWidget> // Required for inheritance

class CodeEditor;
class SyntaxStyle;

/**
 * @brief Class, that describes line number area widget.
 */
class LineNumberArea : public QWidget
{
    Q_OBJECT

public:

    /**
     * @brief Constructor.
     * @param parent Pointer to parent QTextEdit widget.
     */
    explicit LineNumberArea(CodeEditor* parent=nullptr);

    // Disable copying
    LineNumberArea(const LineNumberArea&) = delete;
    LineNumberArea& operator=(const LineNumberArea&) = delete;

    /**
     * @brief Overridden method for getting line number area
     * size.
     */
    QSize sizeHint() const override;

    /**
     * @brief Method for setting syntax style object.
     * @param style Pointer to syntax style.
     */
    void setSyntaxStyle(SyntaxStyle* style);

    /**
     * @brief Method for getting syntax style.
     * @return Pointer to syntax style.
     */
    SyntaxStyle* syntaxStyle() const;

protected:
    void paintEvent(QPaintEvent* event) override;

private:

    SyntaxStyle* m_syntaxStyle;

    CodeEditor* m_codeEditParent;

};

