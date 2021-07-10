#include "main_window.h"

#include <QTimer>
#include <QLineEdit>
#include <QTextEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>

#include "ptx_generator.h"

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

  QObject::connect(cuda->document(), &QTextDocument::contentsChanged, this, &MainWindow::reset_timer);
  QObject::connect(options, &QLineEdit::textChanged, this, &MainWindow::reset_timer);
  QObject::connect(timer, &QTimer::timeout, this, &MainWindow::regen_ptx);

  reset_timer();
}

void MainWindow::reset_timer()
{
  timer->start(500);
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
