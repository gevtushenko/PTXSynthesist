#include "main_window.h"

#include <QTimer>
#include <QToolBar>
#include <QLineEdit>
#include <QTextEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>

#include "ptx_generator.h"
#include "ptx_interpreter.h"
#include "cuda_syntax_highlighter.h"

MainWindow::MainWindow()
  : options(new QLineEdit())
  , cuda(new QTextEdit())
  , ptx(new QTextEdit())
  , timer(new QTimer())
{
  options->setText("-lineinfo, --gpu-architecture=compute_86");

  cuda->setAcceptRichText(false);
  cuda->setPlainText("__global__ void kernel(int *data)\n"
                     "{\n"
                     "    data[0] = 42;\n"
                     "}");

  QHBoxLayout *h_layout = new QHBoxLayout();
  h_layout->addWidget(cuda);
  h_layout->addWidget(ptx);

  QVBoxLayout *v_layout = new QVBoxLayout();
  v_layout->addWidget(options);
  v_layout->addLayout(h_layout);

  QWidget *central_widget = new QWidget();
  central_widget->setLayout(v_layout);

  setCentralWidget(central_widget);

  timer->setSingleShot(true);

  tool_bar = addToolBar("Interpreter");
  QWidget* spacer = new QWidget();
  spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  tool_bar->addWidget(spacer);

  QAction *run_action = new QAction(QIcon(":/icons/play.png"), "Run", this);
  QAction *interpret_action = new QAction(QIcon(":/icons/bug.png"), "Interpret", this);

  tool_bar->addAction(run_action);
  tool_bar->addAction(interpret_action);
  tool_bar->setMovable(false);

  QObject::connect(interpret_action, &QAction::triggered, this, &MainWindow::interpret);

  QObject::connect(cuda->document(), &QTextDocument::contentsChanged, this, &MainWindow::reset_timer);
  QObject::connect(options, &QLineEdit::textChanged, this, &MainWindow::reset_timer);
  QObject::connect(timer, &QTimer::timeout, this, &MainWindow::regen_ptx);

  QFont jet_brains_mono = QFont("JetBrains Mono", 12);
  cuda->setFont(jet_brains_mono);
  options->setFont(jet_brains_mono);

  ptx->setFont(jet_brains_mono);
  ptx->setReadOnly(true);

  CUDASyntaxHighlighter *cuda_syntax_highlighter = new CUDASyntaxHighlighter(cuda->document());

  setStyleSheet("QTextEdit {"
                " background-color: #282a36; "
                " color: #F8F8F2; "
                "}"
                ""
                "QToolBar {"
                " background-color: #414450;"
                "}"
                ""
                "QToolButton {"
                " max-width: 15px; "
                " max-height: 15px; "
                " margin: 8px 4px; "
                "}"
                ""
                "QToolButton:hover {"
                " background-color: #282a36; "
                "}"
                ""
                "QLineEdit {"
                " background-color: #282a36; "
                " color: #F8F8F2; "
                "}"
                ""
                "QScrollBar:vertical {"
                "    padding: 0px 0px 0px 0px;"
                "    background: #282a36;"
                "}"
                "QScrollBar::handle:vertical {"
                "    background: #4c4d56;"
                "}"
                ""
                "QMainWindow {"
                " background-color: #414450; "
                "}");

  reset_timer();
}

void MainWindow::interpret()
{
  PTXInterpreter interpreter;

  interpreter.interpret(ptx->toPlainText().toStdString());
}

void MainWindow::reset_timer()
{
  timer->start(1000);
}

void MainWindow::regen_ptx()
{
  std::string cuda_source = cuda->toPlainText().toStdString();

  QStringList options_list = options->text().split(',');
  std::vector<std::string> str_options_list(options_list.size());
  std::vector<const char *> c_str_options_list(options_list.size());

  for (int i = 0; i < static_cast<int>(c_str_options_list.size()); i++)
  {
    str_options_list[i] = options_list[i].toStdString();
    c_str_options_list[i] = str_options_list[i].c_str();
  }

  PTXGenerator generator;
  std::optional<PTXCode> ptx_code = generator.gen(cuda_source.c_str(), c_str_options_list);

  if (ptx_code)
  {
    ptx->setPlainText(ptx_code->get_ptx());
  }
  else
  {
    ptx->setPlainText("Compilation error");
  }
}
